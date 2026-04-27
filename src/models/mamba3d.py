import torch
import torch.nn as nn
from monai.networks.blocks import UnetResBlock


class BoundaryAwareEnhancementModule(nn.Module):
    """Boundary-Aware Enhancement Module (BAEM).

    The module predicts a 1-channel edge attention map and applies residual
    boundary enhancement: F_enhanced = F + F * E.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.edge_head = nn.Conv3d(in_channels, 1, kernel_size=1, bias=True)
        self.activation = nn.Sigmoid()

    def forward(self, features: torch.Tensor):
        edge_map = self.activation(self.edge_head(features))
        enhanced_features = features + features * edge_map
        return enhanced_features, edge_map


class MambaLayer(nn.Module):
    """
    核心 Mamba 层：通过张量展平和非线性门控，
    模拟 State Space Model (SSM) 的线性复杂度序列扫描过程。
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # 模拟 SSM 内部的状态扩展
        self.ssm = nn.Linear(dim, dim * 2) 
        self.out_proj = nn.Linear(dim * 2, dim)

    def forward(self, x):
        # x shape: (B, C, D, H, W) -> 转换为序列 (B, L, C)
        B, C, D, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        
        # 层归一化与序列扫描模拟
        x_norm = self.norm(x_flat)
        x_ssm = torch.nn.functional.silu(self.ssm(x_norm))
        x_out = self.out_proj(x_ssm)
        
        # 残差连接并恢复 3D 空间维度
        out_flat = x_flat + x_out
        return out_flat.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

class Mamba3D(nn.Module):
    """
    Mamba 3D 混合网络架构
    """
    def __init__(self, in_channels=4, out_channels=5):
        super().__init__()
        # 浅层特征编码
        self.enc = UnetResBlock(
            spatial_dims=3, in_channels=in_channels, out_channels=32, 
            kernel_size=3, stride=1, norm_name="instance"
        )
        
        # 深层长程依赖建模 (Mamba Bottleneck)
        self.mamba = MambaLayer(dim=32)
        
        # 浅层特征解码
        self.dec = UnetResBlock(
            spatial_dims=3, in_channels=32, out_channels=32, 
            kernel_size=3, stride=1, norm_name="instance"
        )

        self.baem = BoundaryAwareEnhancementModule(in_channels=32)
        
        # 分割预测头
        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.enc(x)
        x = self.mamba(x)
        x = self.dec(x)

        enhanced_features, edge_prediction = self.baem(x)
        segmentation_output = self.final_conv(enhanced_features)
        return segmentation_output, edge_prediction

def build_mamba3d(in_channels=4, out_channels=5, device="cpu"):
    """
    实例化并返回 Mamba3D 模型
    """
    return Mamba3D(in_channels, out_channels).to(device)