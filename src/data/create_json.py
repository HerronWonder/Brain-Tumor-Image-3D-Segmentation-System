import os
import glob
import json
import random

def create_brats_json(data_dir, out_json_path, seed=42):
    random.seed(seed)
    
    # 匹配 All 文件夹下所有的病人文件夹
    patient_folders = sorted(glob.glob(os.path.join(data_dir, "BraTS2021_*")))
    
    all_data = []
    for folder in patient_folders:
        pid = os.path.basename(folder) # 例如: BraTS2021_00000
        
        # 核心：只记录相对路径。这样无论项目怎么换电脑，JSON都不会失效
        image_paths = [
            os.path.join(pid, f"{pid}_t1.nii.gz"),
            os.path.join(pid, f"{pid}_t1ce.nii.gz"),
            os.path.join(pid, f"{pid}_t2.nii.gz"),
            os.path.join(pid, f"{pid}_flair.nii.gz")
        ]
        label_path = os.path.join(pid, f"{pid}_seg.nii.gz")
        
        all_data.append({"image": image_paths, "label": label_path})
        
    # 打乱并划分 8:1:1
    random.shuffle(all_data)
    total = len(all_data)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    
    split_dict = {
        "train": all_data[:train_end],
        "val": all_data[train_end:val_end],
        "test": all_data[val_end:]
    }
    
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(split_dict, f, indent=4)
        
    print(f"✅ dataset.json 生成完毕！共处理 {total} 个病例。")
    print(f"Train: {len(split_dict['train'])} | Val: {len(split_dict['val'])} | Test: {len(split_dict['test'])}")

if __name__ == "__main__":
    # 根据你的绝对路径进行替换
    DATA_DIR = "../DataSets/brats-2021-task/All/"
    JSON_OUT = "../DataSets/brats-2021-task/dataset.json"
    create_brats_json(DATA_DIR, JSON_OUT)