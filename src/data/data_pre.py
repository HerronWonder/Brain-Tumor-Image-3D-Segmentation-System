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
    SpatialPadd
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
            SpatialPadd(keys=keys, spatial_size=(128, 128, 128)),
            RandCropByPosNegLabeld(
                keys=keys,
                label_key="label",
                spatial_size=(128, 128, 128),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            )]
        )

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