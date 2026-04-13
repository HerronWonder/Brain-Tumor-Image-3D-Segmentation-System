import os
import sys
import torch
import numpy as np
import nibabel as nib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monai.inferers import sliding_window_inference
from models.unet3d import build_unet3d
from scripts.visualize import build_infer_transforms

def run_medical_inference(patient_dict, weights_path, output_dir, device_str="cpu"):
    device = torch.device(device_str)
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载模型
    model = build_unet3d(in_channels=4, out_channels=5, device=device)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights missing at: {weights_path}")
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

       # 2. 数据预处理
    infer_transforms = build_infer_transforms(has_label=False)
    test_data = infer_transforms(patient_dict)
    input_tensor = test_data["image"].unsqueeze(0).to(device)

    # 背景掩码：任一模态非零即视为前景（避免归一化后正负抵消）
    bg_mask = (test_data["image"] != 0).any(dim=0).cpu().numpy()

    # 3. 滑动窗口推理
    with torch.no_grad():
        outputs = sliding_window_inference(
            inputs=input_tensor,
            roi_size=(128, 128, 128),
            sw_batch_size=4,
            predictor=model
        )
        pred_vol = torch.argmax(outputs, dim=1).cpu().numpy()[0]

    # 强制规则：原图为 0 的位置预测必须为 0
    pred_vol[~bg_mask] = 0

    # ================= 新增：临床物理体积计算 =================
    # 因为经过了 Isotropic Resampling (1x1x1 mm)，1 voxel = 1 mm^3 = 0.001 cm^3
    voxel_vol_cm3 = 0.001
    
    vol_necrotic = float(np.sum(pred_vol == 1)) * voxel_vol_cm3
    vol_edema = float(np.sum(pred_vol == 2)) * voxel_vol_cm3
    vol_enhancing = float(np.sum(pred_vol == 4)) * voxel_vol_cm3
    total_vol = vol_necrotic + vol_edema + vol_enhancing
    
    metrics = {
        "necrotic_cm3": round(vol_necrotic, 2),
        "edema_cm3": round(vol_edema, 2),
        "enhancing_cm3": round(vol_enhancing, 2),
        "total_cm3": round(total_vol, 2)
    }
    # ==========================================================

    # 4. 生成 NIfTI 文件
    original_nii = nib.load(patient_dict["image"][1]) 
    mask_nii = nib.Nifti1Image(pred_vol.astype(np.uint8), original_nii.affine)
    output_path = os.path.join(output_dir, "pred_mask.nii.gz")
    nib.save(mask_nii, output_path)
    
    # 注意：这里不仅返回路径，还把体积指标一起返回了
    return output_path, metrics