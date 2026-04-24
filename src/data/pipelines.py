import json
import os
from typing import Dict, List

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    Spacingd,
    SpatialPadd,
)


def build_brats_transforms(is_train: bool = True):
    keys = ["image", "label"]
    transforms = [
        LoadImaged(keys=keys, reader="NibabelReader"),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=keys, source_key="image"),
    ]

    if is_train:
        transforms.extend(
            [
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
                ),
            ]
        )

    transforms.append(EnsureTyped(keys=keys))
    return Compose(transforms)


def build_infer_transforms(has_label: bool = False):
    keys = ["image", "label"] if has_label else ["image"]
    mode = ("bilinear", "nearest") if has_label else ("bilinear",)

    return Compose(
        [
            LoadImaged(keys=keys, reader="NibabelReader"),
            EnsureChannelFirstd(keys=keys),
            Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=mode),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=keys),
        ]
    )


def get_brats_data_dicts(data_dir: str, json_path: str, split: str = "train") -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        dataset_split = json.load(f)

    data_dicts: List[Dict] = []
    for item in dataset_split.get(split, []):
        image_paths = [os.path.join(data_dir, p) for p in item["image"]]
        label_path = os.path.join(data_dir, item["label"])

        if all(os.path.exists(p) for p in image_paths) and os.path.exists(label_path):
            data_dicts.append({"image": image_paths, "label": label_path})

    return data_dicts