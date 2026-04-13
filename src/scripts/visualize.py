import os
import sys
import torch
import numpy as np
import warnings
import matplotlib
import argparse

matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, NormalizeIntensityd, EnsureTyped
)
from monai.inferers import sliding_window_inference
from models.unet3d import build_unet3d
from models.mamba3d import build_mamba3d

# ==========================================
# 全局配置区
# ==========================================
DATA_DIRECTORY = "../../../DataSets/brats-2021-task/All/"
PATIENT_INDEX = 1  
OUTPUT_NAME = "inference_result.png" 
MODEL_TYPE = "mamba" 
WEIGHTS_PATH = "weights/mamba_unet.pth" 
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="3D Medical Image Inference and Visualization")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY)
    parser.add_argument("--patient_idx", type=int, default=PATIENT_INDEX)
    parser.add_argument("--weights", type=str, default=WEIGHTS_PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_NAME)
    parser.add_argument("--model_type", type=str, default=MODEL_TYPE, choices=["unet3d", "mamba"])
    return parser.parse_args()

def build_infer_transforms(has_label=False):
    """构建动态验证/推理数据预处理管线"""
    keys = ["image", "label"] if has_label else ["image"]
    mode = ("bilinear", "nearest") if has_label else ("bilinear",)
    
    return Compose([
        LoadImaged(keys=keys, reader="NibabelReader"),
        EnsureChannelFirstd(keys=keys),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=mode),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=keys)
    ])

def get_center_slices(reference_vol):
    """寻找包含最多肿瘤/前景的切片坐标，或返回几何中心"""
    coords = np.where(reference_vol > 0)
    if len(coords[0]) > 0:
        return int(np.median(coords[0])), int(np.median(coords[1])), int(np.median(coords[2]))
    return reference_vol.shape[0]//2, reference_vol.shape[1]//2, reference_vol.shape[2]//2

def save_slice_plot(img_vol, pred_vol, gt_vol, slices, pid_name, save_path):
    """核心绘图引擎：负责 3x3 或 2x3 多视角切片渲染"""
    x, y, z = slices
    has_label = gt_vol is not None
    rows = 3 if has_label else 2
    
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    fig.suptitle(f"3D Medical Image Segmentation Results\nPatient: {pid_name}", fontsize=16)

    views = [(z, 2, "Axial"), (y, 1, "Coronal"), (x, 0, "Sagittal")]

    for col, (idx, axis, title) in enumerate(views):
        # 优雅地提取指定轴向的切片并旋转
        img2d = np.rot90(np.take(img_vol, idx, axis=axis))
        pred2d = np.rot90(np.take(pred_vol, idx, axis=axis))
        gt2d = np.rot90(np.take(gt_vol, idx, axis=axis)) if has_label else None

        # 原图层
        axes[0, col].imshow(img2d, cmap="gray")
        axes[0, col].set_title(f"T1ce MRI - {title}")
        axes[0, col].axis("off")

        row_offset = 1
        # 真实标签层 (如果有)
        if has_label:
            axes[row_offset, col].imshow(img2d, cmap="gray")
            axes[row_offset, col].imshow(np.ma.masked_where(gt2d == 0, gt2d), cmap="jet", alpha=0.5, vmin=0, vmax=4)
            axes[row_offset, col].set_title("Ground Truth (Validation)")
            axes[row_offset, col].axis("off")
            row_offset += 1

        # 模型预测层
        axes[row_offset, col].imshow(img2d, cmap="gray")
        axes[row_offset, col].imshow(np.ma.masked_where(pred2d == 0, pred2d), cmap="jet", alpha=0.5, vmin=0, vmax=4)
        axes[row_offset, col].set_title("AI Model Prediction")
        axes[row_offset, col].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def run_inference(model, patient_dict, device_str, save_path):
    """统筹推理流程的高层接口 """
    device = torch.device(device_str)
    has_label = 'label' in patient_dict
    
    # 解析病人ID
    if has_label:
        pid_name = os.path.basename(patient_dict['label']).split('_seg')[0]
    else:
        pid_name = os.path.basename(patient_dict['image'][0]).split('_t1')[0]

    # 预处理与加载
    test_data = build_infer_transforms(has_label)(patient_dict)
    input_tensor = test_data["image"].unsqueeze(0).to(device)

    # 模型推理
    with torch.no_grad():
        outputs = sliding_window_inference(input_tensor, (128, 128, 128), 4, model)
        pred_vol = torch.argmax(outputs, dim=1).cpu().numpy()[0]

    # 解析数据用于渲染
    img_vol = input_tensor.cpu().numpy()[0, 1] 
    gt_vol = test_data["label"].numpy()[0] if has_label else None
    
    # 定位与渲染
    slices = get_center_slices(gt_vol if has_label else pred_vol)
    save_slice_plot(img_vol, pred_vol, gt_vol, slices, pid_name, save_path)

    return save_path

def main():
    args = parse_args()
    from data.data_pre import get_brats_data_dicts

    print("-" * 60)
    print(f"[TEST] Initializing Visualization Engine (Model: {args.model_type})")
    
    val_files = get_brats_data_dicts(args.data_dir)
    if not val_files:
        print("[ERROR] Dataset not found.")
        return

    try:
        test_patient = val_files[args.patient_idx]
    except IndexError:
        print(f"[ERROR] Patient index out of bounds.")
        return

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_script_dir, args.weights)
    save_fig_path = os.path.join(os.path.dirname(current_script_dir), "outputs", args.output)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # 模型实例化与权重加载隔离在主函数中
    if args.model_type == "unet3d":
        model = build_unet3d(in_channels=4, out_channels=5, device=device_str)
    elif args.model_type == "mamba":
        model = build_mamba3d(in_channels=4, out_channels=5, device=device_str)
    else:
        print("[WARNING] Unknown model, fallback to U-Net.")
        model = build_unet3d(in_channels=4, out_channels=5, device=device_str)

    if not os.path.exists(weights_path):
        print(f"[ERROR] Model weights not found at: {weights_path}")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device_str))
    model.eval()

    print("[TEST] Running inference and generating plots...")
    try:
        final_path = run_inference(model, test_patient, device_str, save_fig_path)
        print(f"[SUCCESS] Visualization saved to: {final_path}")
    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")
    print("-" * 60)

if __name__ == "__main__":
    main()