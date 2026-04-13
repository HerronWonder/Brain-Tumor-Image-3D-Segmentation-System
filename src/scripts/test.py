import os
import sys
import time
import torch
import warnings
import argparse
import numpy as np

# 动态将项目根目录加入环境变量
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import build_train_loader
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

warnings.filterwarnings("ignore")

# ==========================================
# 默认配置区
# ==========================================
DATA_DIRECTORY = "../../../DataSets/brats-2021-task/All/"
JSON_PATH = "../../../DataSets/brats-2021-task/dataset.json"

def get_model(model_name, device):
    """根据参数实例化不同的模型架构"""
    if model_name == "unet":
        from models.unet3d import build_unet3d
        return build_unet3d(in_channels=4, out_channels=5, device=device)
    elif model_name == "mamba":
        from models.mamba3d import build_mamba3d
        return build_mamba3d(in_channels=4, out_channels=5, device=device)
    else:
        raise ValueError(f"不支持的模型架构: {model_name}")

def parse_args():
    parser = argparse.ArgumentParser(description="3D Medical Image Inference & Testing")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY)
    parser.add_argument("--json_path", type=str, default=JSON_PATH)
    
    # 核心参数：指定测试哪个模型，以及加载哪个权重
    parser.add_argument("--model", type=str, choices=["unet", "mamba"], required=True, 
                        help="选择要测试的模型架构 (unet 或 mamba)")
    parser.add_argument("--weight_path", type=str, required=True, 
                        help="训练好的 .pth 权重文件路径")
    return parser.parse_args()

def main():
    args = parse_args()

    print("\n" + "="*75)
    print(f"[TESTING] 3D Segmentation Architecture : {args.model.upper()}")
    print("="*75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Computation Device          : {device}")

    # ==================================
    # 1. 加载测试集 (Test Split)
    # ==================================
    # ⚠️ 测试时必须跑全量数据，所以 mini_dataset_size=None
    test_loader, test_count = build_train_loader(
        data_dir=args.data_dir, 
        json_path=args.json_path, 
        split="test", 
        batch_size=1,     # 测试时 batch_size 强制为 1
        mini_dataset_size=None
    )
    print(f"[DATA]   Loaded Test Set             : {test_count} patients")

    # ==================================
    # 2. 构建模型并加载权重
    # ==================================
    model = get_model(args.model, device)
    
    if os.path.exists(args.weight_path):
        model.load_state_dict(torch.load(args.weight_path, map_location=device))
        print(f"[MODEL]  Successfully loaded weights from: {args.weight_path}")
    else:
        raise FileNotFoundError(f"找不到权重文件: {args.weight_path}")
    
    model.eval()

    # ==================================
    # 3. 初始化推理器与评估指标
    # ==================================
    # 滑动窗口推理：roi_size 必须与你训练时 Crop 的空间大小一致 (128x128x128)
    inferer = SlidingWindowInferer(roi_size=(128, 128, 128), sw_batch_size=4, overlap=0.5)
    
    # BraTS 通常有 5 个类别 (包括背景 0)
    # 计算 Dice 时，模型输出需要进行 Argmax 然后转 One-Hot，真实标签也需要转 One-Hot
    post_pred = AsDiscrete(argmax=True, to_onehot=5)
    post_label = AsDiscrete(to_onehot=5)
    
    # include_background=False 表示计算平均 Dice 时不把广袤的黑色背景算进去
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    print("\n" + "-"*75)
    print("Starting Testing Loop...")
    print("-" * 75)

    total_start_time = time.time()

    # ==================================
    # 4. 测试主循环
    # ==================================
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_inputs = test_data["image"].to(device)
            test_labels = test_data["label"].to(device)

            # 1. 前向传播 (使用滑动窗口)
            test_outputs = inferer(test_inputs, model)
            
            # 2. 后处理：转换为 One-Hot 格式以计算 Dice
            val_outputs = [post_pred(i) for i in test_outputs]
            val_labels = [post_label(i) for i in test_labels]
            
            # 3. 压入当前 batch 的结果用于后续统计
            dice_metric(y_pred=val_outputs, y=val_labels)
            
            # 打印当前样本的进度
            print(f"[INFER]  Processed patient {i + 1:03d}/{test_count}...")

    # ==================================
    # 5. 汇总与输出报告
    # ==================================
    # 聚合并计算所有测试样本的平均 Dice 分数
    mean_dice = dice_metric.aggregate().item()
    
    # 重置 metric 状态，释放内存
    dice_metric.reset()

    total_duration = time.time() - total_start_time
    print("-" * 75)
    print(f"Testing Completed in {total_duration:.2f}s.")
    print("\n" + "★"*75)
    print(f"Final Test Metrics for [{args.model.upper()}]:")
    print(f"-> Mean Dice Score (excluding background) : {mean_dice:.4f}")
    print("★"*75 + "\n")

if __name__ == "__main__":
    main()