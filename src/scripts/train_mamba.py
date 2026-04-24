import os
import sys
import time
import torch
import warnings
import wandb
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import build_train_loader
from models.mamba3d import build_mamba3d
from utils.losses import build_loss_function
from torch.optim import Adam

warnings.filterwarnings("ignore")

# ==========================================
# 常用超参数配置区
# ==========================================
DATA_DIRECTORY = "../../../dataset/All/"
JSON_PATH = "../../../dataset/dataset.json"
EPOCH_NUM = 95
LEARNING_RATE = 1e-4
RESUME_WEIGHT = "weights/mamba_unet_last.pth" # 改为读取 last

def parse_args():
    parser = argparse.ArgumentParser(description="3D Mamba-UNet Training Script")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY)
    parser.add_argument("--json_path", type=str, default=JSON_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCH_NUM)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--resume_weight", type=str, default=RESUME_WEIGHT)
    return parser.parse_args()

def main():
    args = parse_args()

    print("\n" + "="*75)
    print("[PHASE 2] 3D Mamba Architecture Training (With Validation)")
    print("="*75)

    run = wandb.init(
        entity="herron_wonder",
        project="FinalDesign",
        id="mamba_midterm",             
        resume="allow",     
        config={
            "learning_rate": args.lr,
            "architecture": "mamba_unet",
            "dataset": "brats-2021-task",
            "epochs_per_run": args.epochs,
            "resume_weight": args.resume_weight
        },
    )

    # WandB 断点续传起始 Epoch
    start_epoch = run.step if run.resumed else 0
    print(f"[WANDB]  Resumed: {run.resumed} | Starting from Epoch: {start_epoch}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Computation Device Allocated : {device}")

    # ==================================
    # 1. 数据加载 (Train & Val)
    # ==================================
    train_loader, train_count = build_train_loader(
        data_dir=args.data_dir,
        json_path=args.json_path,
        split="train",
        batch_size=1,
        mini_dataset_size=5 # 正式训练时记得改为 None
    )
    
    val_loader, val_count = build_train_loader(
        data_dir=args.data_dir,
        json_path=args.json_path,
        split="val",
        batch_size=1,
        mini_dataset_size=5 # 正式跑改为 None
    )
    print(f"[DATA]   Loaded Mini-Dataset            : {train_count} patients (Train)")
    print(f"[DATA]   Loaded Val Set                 : {val_count} patients (Val)")

    # ==================================
    # 2. 模型与优化器构建
    # ==================================
    model = build_mamba3d(in_channels=4, out_channels=5, device=device)
    print("[MODEL]  Architecture Setup             : 3D Mamba Hybrid")

    if args.resume_weight is not None:
        if os.path.exists(args.resume_weight):
            model.load_state_dict(torch.load(args.resume_weight, map_location=device))
            print(f"[MODEL]  Successfully loaded weights from: {args.resume_weight}")
        else:
            print(f"[WARNING] Weight file not found at {args.resume_weight}. Training from scratch.")

    loss_function = build_loss_function()
    optimizer = Adam(model.parameters(), args.lr)
    
    print(f"[CONFIG] Loss Function Strategy       : Dice + Cross Entropy")
    print(f"[CONFIG] Additional Epochs to Train   : {args.epochs}")

    # ==================================
    # 3. 训练主循环
    # ==================================
    print("\n" + "-"*75)
    print("Starting Model Training Loop...")
    print("-" * 75)

    total_start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_start_time = time.time()
        
        # 阶段 A: 训练
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / step
        epoch_duration = time.time() - epoch_start_time
        print(f"[TRAIN]  Epoch {epoch + 1:03d} | Duration: {epoch_duration:.2f}s | Train Loss: {avg_train_loss:.4f}")
        
        # 阶段 B: 验证
        model.eval()
        val_epoch_loss = 0
        val_step = 0
        
        with torch.no_grad():
            for val_data in val_loader:
                val_step += 1
                val_inputs = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)
                
                val_outputs = model(val_inputs)
                v_loss = loss_function(val_outputs, val_labels)
                val_epoch_loss += v_loss.item()
                
        avg_val_loss = val_epoch_loss / val_step
        print(f"[VAL]    Epoch {epoch + 1:03d} | Val Loss: {avg_val_loss:.4f}")
        
        # 将 train 和 val loss 同步记录到 wandb
        run.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss}, step=epoch + 1)

        # 阶段 C: 权重持久化
        weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
        os.makedirs(weights_dir, exist_ok=True)
        
        last_save_path = os.path.join(weights_dir, "mamba_unet_last.pth")
        torch.save(model.state_dict(), last_save_path)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_save_path = os.path.join(weights_dir, "mamba_unet_best.pth")
            torch.save(model.state_dict(), best_save_path)
            print(f"🌟 [SAVE] New Best Model saved at epoch {epoch + 1} with Val Loss: {best_val_loss:.4f}")

    total_duration = time.time() - total_start_time
    print("-" * 75)
    print(f"Training Batch Completed in {total_duration:.2f}s.")
    run.finish()

if __name__ == "__main__":
    main()