import torch
from monai.networks.nets import UNet

def build_unet3d(in_channels=4, out_channels=5, device="cpu"):
    """
    构建并返回 3D U-Net 基线模型
    """
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    ).to(device)
    return model