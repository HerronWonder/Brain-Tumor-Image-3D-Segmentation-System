import os
import sys
import time
import torch
import warnings
import wandb
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import build_train_loader
from models.unet3d import build_unet3d
from utils.losses import build_loss_function
from torch.optim import Adam
from monai.inferers import SlidingWindowInferer

warnings.filterwarnings("ignore")

# ==========================================
# 常用超参数配置区
# ==========================================
DATA_DIRECTORY = "../../../DataSets/brats-2021-task/All/"
JSON_PATH = "../../../DataSets/brats-2021-task/dataset.json"
EPOCH_NUM = 100
LEARNING_RATE = 1e-4
RESUME_WEIGHT = "weights/baseline_unet_last.pth" 

def parse_args():
    parser = argparse.ArgumentParser(description="3D U-Net Training Script (Multi-GPU Adaptive)")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY)
    parser.add_argument("--json_path", type=str, default=JSON_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCH_NUM)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--resume_weight", type=str, default=RESUME_WEIGHT)
    parser.add_argument("--bs_per_gpu", type=int, default=1, help="Batch size per GPU")
    return parser.parse_args()

def save_model(model, save_path):
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)

def main():
    args = parse_args()

    print("\n" + "="*75)
    print("3D Segmentation Architecture Training (unet)")
    print("="*75)

    # 1. 环境与自适应 Batch Size 计算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 🌟 核心逻辑：计算真实的总 Batch Size
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    total_batch_size = args.bs_per_gpu * gpu_count
    
    print(f"[SYSTEM] 探测到可见 GPU 数量   : {gpu_count}")
    print(f"[CONFIG] 单卡 Batch Size 设定 : {args.bs_per_gpu}")
    print(f"[CONFIG] 实际运行总 Batch Size  : {total_batch_size} ({args.bs_per_gpu} × {gpu_count})")

    run = wandb.init(
        entity="herron_wonder",
        project="FinalDesign",
        id="baseline_1.0.0",                
        resume="allow",     
        config={
            "learning_rate": args.lr,
            "architecture": "baseline_unet",
            "dataset": "brats-2021-task",
            "epochs": args.epochs,
            "bs_per_gpu": args.bs_per_gpu,
            "total_batch_size": total_batch_size, # 将真实的 BS 记录进 WandB
            "gpu_count": gpu_count
        },
    )

    # 2. 数据加载 (传入计算好的 total_batch_size)
    # ⚠️ 注意：正式训练时务必将 mini_dataset_size=5 改为 None
    train_loader, train_count = build_train_loader(
        data_dir=args.data_dir, 
        json_path=args.json_path, 
        split="train", 
        batch_size=total_batch_size,  # 🌟 使用自适应 BS
        mini_dataset_size=5 
    )

    val_loader, val_count = build_train_loader(
        data_dir=args.data_dir, 
        json_path=args.json_path, 
        split="val", 
        batch_size=total_batch_size,  # 🌟 使用自适应 BS
        mini_dataset_size=5 
    )
    print(f"[DATA]   Loaded Mini-Dataset            : {train_count} patients (Train)")
    print(f"[DATA]   Loaded Val Set                 : {val_count} patients (Val)")

    # 3. 模型构建与多卡分发
    model = build_unet3d(in_channels=4, out_channels=5, device=device)

    if gpu_count > 1:
        print(f"[SYSTEM] 自动开启 DataParallel 多卡并行加速！")
        model = torch.nn.DataParallel(model)

    model.to(device)
    print("[MODEL]  Architecture Setup             : 3D U-Net")

    # 断点续训逻辑
    if args.resume_weight is not None:
        if os.path.exists(args.resume_weight):
            model.load_state_dict(torch.load(args.resume_weight, map_location=device))
            print(f"[MODEL]  Successfully loaded weights from: {args.resume_weight}")
        else:
            print(f"[WARNING] Weight file not found at {args.resume_weight}. Training from scratch.")

    # 4. 优化器与损失函数
    loss_function = build_loss_function()
    optimizer = Adam(model.parameters(), args.lr)
    
    print(f"[CONFIG] Loss Function Strategy       : Dice + Cross Entropy")
    print(f"[CONFIG] Target Training Epochs       : {args.epochs}")

    # 5. 训练主循环
    print("\n" + "-"*75)
    print("Starting Model Training Loop...")
    print("-" * 75)

    total_start_time = time.time()
    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # ==================================
        # 阶段 A: 训练
        # ==================================
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
        print(f"[TRAIN]  Epoch {epoch + 1:02d}/{args.epochs:02d} | Duration: {epoch_duration:.2f}s | Train Loss: {avg_train_loss:.4f}")

        # ==================================
        # 阶段 B: 验证
        # ==================================
        model.eval()
        val_epoch_loss = 0
        val_step = 0
        
        val_inferer = SlidingWindowInferer(roi_size=(128, 128, 128), sw_batch_size=4, overlap=0.5)
        
        with torch.no_grad():
            for val_data in val_loader:
                val_step += 1
                val_inputs = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)
                
                val_outputs = val_inferer(val_inputs, model)
                v_loss = loss_function(val_outputs, val_labels)
                val_epoch_loss += v_loss.item()
                
        avg_val_loss = val_epoch_loss / val_step
        print(f"[VAL]    Epoch {epoch + 1:02d} | Val Loss: {avg_val_loss:.4f}")
        run.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # ==================================
        # 阶段 C: 权重持久化
        # ==================================
        weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
        os.makedirs(weights_dir, exist_ok=True)
        
        last_save_path = os.path.join(weights_dir, "baseline_unet_last.pth")
        save_model(model, last_save_path)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_save_path = os.path.join(weights_dir, "baseline_unet_best.pth")
            save_model(model, best_save_path)
            print(f"🌟 [SAVE] New Best Model saved at epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")

    total_duration = time.time() - total_start_time
    print("-" * 75)
    print(f"Training Completed in {total_duration:.2f}s.")
    run.finish()

if __name__ == "__main__":
    main()