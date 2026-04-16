import os
import glob
import torch
import warnings
import json


# 忽略 MONAI 的底层版本兼容性警告，保证控制台输出纯净
warnings.filterwarnings("ignore")

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    EnsureTyped,
    SpatialPadd,
    RandFlipd,
    RandAffined,
    RandShiftIntensityd
)
from monai.data import Dataset, DataLoader

def get_brats_transforms(is_train=True):
    keys = ["image", "label"]
    
    transforms = [
        LoadImaged(keys=keys, reader="NibabelReader"),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys, 
            pixdim=(1.0, 1.0, 1.0), 
            mode=("bilinear", "nearest")
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=keys, source_key="image")
    ]
    
    if is_train:
        transforms.extend([
            # 1. 安全垫：保证尺寸足够裁剪
            SpatialPadd(keys=keys, spatial_size=(128, 128, 128)),
            
            # 2. 核心裁剪：切出 4 个 128x128x128 的块
            RandCropByPosNegLabeld(
                keys=keys,
                label_key="label",
                spatial_size=(128, 128, 128),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            
            # ================= 新增：真正的空间与像素级数据增强 =================
            # 3. 随机镜像翻转 (概率 50%，分别沿 X, Y, Z 轴)
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            
            # 4. 随机仿射变换 (轻微的旋转、缩放、平移，模拟病人在机器里的轻微乱动)
            RandAffined(
                keys=keys,
                mode=("bilinear", "nearest"),
                prob=0.5,
                spatial_size=(128, 128, 128),
                rotate_range=(0.1, 0.1, 0.1), # 约 5.7 度的旋转
                scale_range=(0.1, 0.1, 0.1)   # 10% 的缩放
            ),
            
            # 5. 随机强度偏移 (模拟不同 MRI 仪器的亮度差异)
            RandShiftIntensityd(
                keys="image", # 注意：只对图像做，绝对不能改变 label 的值
                offsets=0.1,
                prob=0.5,
            )
            # ====================================================================
        ])

    transforms.append(EnsureTyped(keys=keys))
    
    return Compose(transforms)

def get_brats_data_dicts(data_dir, json_path, split="train"):
    """
    通过读取 JSON 文件获取数据集字典
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset_split = json.load(f)
        
    data_dicts = []
    
    # 遍历对应划分（如 "train"）的数据
    for item in dataset_split[split]:
        # 将 JSON 中的相对路径与 data_dir 拼接为绝对路径
        image_paths = [os.path.join(data_dir, p) for p in item["image"]]
        label_path = os.path.join(data_dir, item["label"])
        
        # 保留你的安全检查：确保文件真实存在才加入训练列表
        if all(os.path.exists(p) for p in image_paths) and os.path.exists(label_path):
            data_dicts.append({"image": image_paths, "label": label_path})
            
    return data_dicts

if __name__ == "__main__":
    data_directory = "../../DataSets/brats-2021-task/All/"
    json_path = "../../DataSets/brats-2021-task/dataset.json"

    print("\n" + "="*60)
    print("3D MRI Data Preprocessing Pipeline Initialization")
    print("="*60)
    
    train_files = get_brats_data_dicts(data_directory, json_path, split="train")
    print(f"[INFO] Successfully scanned dataset directory.")
    print(f"[INFO] Found {len(train_files)} valid patient records.")
    
    if len(train_files) > 0:
        print("\n" + "-"*60)
        print("Running Preprocessing Pipeline Test (Batch Size=1, Patches=4)")
        print("-"*60)
        
        train_transforms = get_brats_transforms(is_train=True)
        check_ds = Dataset(data=train_files[:1], transform=train_transforms)
        check_loader = DataLoader(check_ds, batch_size=1)
        
        for check_data in check_loader:
            image, label = check_data["image"], check_data["label"]
            print(f"[OUTPUT] Image Tensor Shape : {image.shape}") 
            print(f"[OUTPUT] Label Tensor Shape : {label.shape}")
            
            img_c0 = image[0, 0] 
            mean_val = img_c0[img_c0 != 0].mean().item()
            print(f"[OUTPUT] Brain Region Mean (Z-Score) : {mean_val:.4f}")
            break
            
        print("\n" + "="*60)
        print("Pipeline is successfully built, aligned, and validated!")
        print("="*60 + "\n")
    else:
        print("Error: Dataset not found. Please check your relative path.")