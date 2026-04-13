from monai.data import Dataset, DataLoader
from .data_pre import get_brats_data_dicts, get_brats_transforms


def build_train_loader(data_dir, json_path, split="train", batch_size=1, mini_dataset_size=5):
    """
    构建数据加载器 (现在既能构建 Train 也能构建 Val)
    """
    data_files = get_brats_data_dicts(data_dir, json_path, split=split)
    

    # # 截取小型数据集用于结构验证
    # if mini_dataset_size is not None and mini_dataset_size > 0:
    #     data_files = data_files[:mini_dataset_size]


    # 注意：如果未来是构建 val_loader，记得写一个 get_brats_transforms(is_train=False) 
    # 把 RandCrop 和翻转等数据增强关掉
    is_train = True if split == "train" else False
    transforms = get_brats_transforms(is_train=is_train)
    
    # 构建 Dataset 和 Loader
    ds = Dataset(data=data_files, transform=transforms)
    # 如果是验证集/测试集，不需要 shuffle
    loader = DataLoader(ds, batch_size=batch_size, shuffle=is_train)
    
    return loader, len(data_files)