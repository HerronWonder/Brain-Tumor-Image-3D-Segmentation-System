import os
import sys
import torch
import numpy as np
import warnings
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

# 动态将项目根目录加入环境变量
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, NormalizeIntensityd, EnsureTyped
)
from monai.inferers import sliding_window_inference
from data.data_pre import get_brats_data_dicts

# 导入你的模型架构
from models.unet3d import build_unet3d
from models.mamba3d import build_mamba3d

# ==========================================
# 默认配置区
# ==========================================
DATA_DIRECTORY = "../../../DataSets/brats-2021-task/All/"
JSON_PATH = "../../../DataSets/brats-2021-task/dataset.json"
PATIENT_INDEX = 0  # 选择用于生成论文插图的病人索引
OUTPUT_NAME = "qualitative_comparison.pdf" # 直接输出PDF以保证LaTeX矢量清晰度

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Figure 3 for Thesis: Model Comparison")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY)
    parser.add_argument("--json_path", type=str, default=JSON_PATH)
    parser.add_argument("--patient_idx", type=int, default=PATIENT_INDEX)
    parser.add_argument("--unet_weights", type=str, required=True, help="Path to 3D U-Net .pth")
    parser.add_argument("--mamba_weights", type=str, required=True, help="Path to Mamba-U-Net .pth")
    parser.add_argument("--output", type=str, default=OUTPUT_NAME)
    return parser.parse_args()

def build_infer_transforms():
    """构建推理数据预处理管线"""
    keys = ["image", "label"]
    return Compose([
        LoadImaged(keys=keys, reader="NibabelReader"),
        EnsureChannelFirstd(keys=keys),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=keys)
    ])

def get_brats_custom_cmap():
    """
    定制 BraTS 颜色映射，严格匹配你的学术描述：
    0: 背景 (透明)
    1: 坏死核心 (Necrotic Core) -> 红色 (Red)
    2: 瘤周水肿 (Edema) -> 绿色 (Green)
    3: (BraTS通常跳过3) -> 透明
    4: 增强肿瘤 (Enhancing Tumor) -> 黄色 (Yellow)
    """
    colors = [
        (0, 0, 0, 0.0),       # 0: Transparent
        (1.0, 0.0, 0.0, 1.0), # 1: Red
        (0.0, 1.0, 0.0, 1.0), # 2: Green
        (0, 0, 0, 0.0),       # 3: Transparent
        (1.0, 1.0, 0.0, 1.0)  # 4: Yellow
    ]
    return mcolors.ListedColormap(colors)

def get_max_tumor_slice(gt_vol):
    """寻找包含肿瘤面积最大的轴状面(Axial)切片索引，最适合做展示"""
    tumor_pixels_per_slice = np.sum(gt_vol > 0, axis=(1, 2))
    best_z = np.argmax(tumor_pixels_per_slice)
    return best_z

def plot_paper_figure(img_slice, gt_slice, unet_slice, mamba_slice, pid_name, save_path):
    """渲染 1x3 论文对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    # fig.suptitle(f"Patient: {pid_name} - Axial View", fontsize=14, y=1.05)
    
    cmap_mask = get_brats_custom_cmap()
    
    # 辅助函数：叠加渲染
    def overlay_mask(ax, bg_img, mask_img, title):
        ax.imshow(bg_img, cmap="gray")
        # 掩码中为0的部分透明显示
        masked_data = np.ma.masked_where(mask_img == 0, mask_img)
        ax.imshow(masked_data, cmap=cmap_mask, alpha=0.6, vmin=0, vmax=4)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
        ax.axis("off")

    # 1. Ground Truth (左)
    overlay_mask(axes[0], img_slice, gt_slice, "Ground Truth")
    
    # 2. 3D U-Net (中)
    overlay_mask(axes[1], img_slice, unet_slice, "3D U-Net")
    
    # 3. Mamba-U-Net (右)
    overlay_mask(axes[2], img_slice, mamba_slice, "Mamba-U-Net")

    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    
    plt.savefig(save_path, format='pdf', bbox_inches='tight', transparent=True)
    plt.close(fig)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"[PAPER FIG] Generating Figure 3 for Thesis")
    print("=" * 60)

    # 1. 加载数据
    val_files = get_brats_data_dicts(args.data_dir, args.json_path, split="test") # 或 "val"
    if not val_files:
        print("[ERROR] Dataset not found.")
        return
    patient_dict = val_files[args.patient_idx]
    pid_name = os.path.basename(patient_dict['label']).split('_seg')[0]
    
    print(f"[DATA] Processing Patient: {pid_name}")
    test_data = build_infer_transforms()(patient_dict)
    input_tensor = test_data["image"].unsqueeze(0).to(device)
    gt_vol = test_data["label"].numpy()[0]

    # 2. 寻找最佳展示切片 (Z轴)
    best_z = get_max_tumor_slice(gt_vol)
    img_slice = np.rot90(input_tensor.cpu().numpy()[0, 1, best_z, :, :]) # 取 T1ce 模态作为背景
    gt_slice = np.rot90(gt_vol[best_z, :, :])

    # 3. 推理 3D U-Net
    print("[MODEL] Running 3D U-Net Inference...")
    unet_model = build_unet3d(in_channels=4, out_channels=5, device=device)
    unet_model.load_state_dict(torch.load(args.unet_weights, map_location=device))
    unet_model.eval()
    with torch.no_grad():
        unet_outputs = sliding_window_inference(input_tensor, (128, 128, 128), 4, unet_model)
        unet_vol = torch.argmax(unet_outputs, dim=1).cpu().numpy()[0]
    unet_slice = np.rot90(unet_vol[best_z, :, :])
    del unet_model # 释放显存
    torch.cuda.empty_cache()

    # 4. 推理 Mamba-U-Net
    print("[MODEL] Running Mamba-U-Net Inference...")
    mamba_model = build_mamba3d(in_channels=4, out_channels=5, device=device)
    mamba_model.load_state_dict(torch.load(args.mamba_weights, map_location=device))
    mamba_model.eval()
    with torch.no_grad():
        mamba_outputs = sliding_window_inference(input_tensor, (128, 128, 128), 4, mamba_model)
        mamba_vol = torch.argmax(mamba_outputs, dim=1).cpu().numpy()[0]
    mamba_slice = np.rot90(mamba_vol[best_z, :, :])

    # 5. 渲染出图
    print(f"[PLOT] Rendering comparative figure to {args.output}...")
    plot_paper_figure(img_slice, gt_slice, unet_slice, mamba_slice, pid_name, args.output)
    print("[SUCCESS] Done!")

if __name__ == "__main__":
    main()