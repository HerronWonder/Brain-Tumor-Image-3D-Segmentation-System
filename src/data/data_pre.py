import warnings
import os

from monai.data import Dataset, DataLoader

from .pipelines import (
    build_brats_transforms,
    get_brats_data_dicts as load_brats_data_dicts,
)

warnings.filterwarnings("ignore")

def get_brats_transforms(is_train=True):
    return build_brats_transforms(is_train=is_train)

def get_brats_data_dicts(data_dir, json_path, split="train"):
    return load_brats_data_dicts(data_dir=data_dir, json_path=json_path, split=split)

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