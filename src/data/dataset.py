import torch

from monai.data import Dataset, DataLoader
from .data_pre import get_brats_data_dicts, get_brats_transforms


def build_train_loader(
    data_dir,
    json_path,
    split="train",
    batch_size=1,
    mini_dataset_size=None,
    num_workers=8,
    pin_memory=None,
):
    data_files = get_brats_data_dicts(data_dir, json_path, split=split)

    if mini_dataset_size is not None and mini_dataset_size > 0:
        data_files = data_files[:mini_dataset_size]

    is_train = split == "train"
    transforms = get_brats_transforms(is_train=is_train)

    ds = Dataset(data=data_files, transform=transforms)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader, len(data_files)