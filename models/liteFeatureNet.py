import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import torch.nn.utils.prune as prune

class LightConvBnReLU6(nn.Module):
    """Lightweight 2D convolution using depthwise separable convolution + BN + ReLU6"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, dilation: int = 1) -> None:
        super(LightConvBnReLU6, self).__init__()
        # Depthwise convolution: 每个输入通道单独卷积
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, dilation=dilation,
                                 groups=in_channels, bias=False)
        # Pointwise convolution: 1×1 卷积，融合通道信息
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = self.bn(x)
        return self.relu6(x)

class LightFeatureNet(nn.Module):
    """
    简化版轻量化 FeatureNet，通过删除部分卷积结构进一步降低模型复杂度，
    并输出多尺度特征：
      - Stage 3 (Deep): [B,64, H/8, W/8]
      - Stage 2 (Intermediate): [B,32, H/4, W/4]
      - Stage 1 (Shallow): [B,16, H/2, W/2]
    """
    def __init__(self):
        super(LightFeatureNet, self).__init__()
        self.conv0 = LightConvBnReLU6(3, 8, kernel_size=3, stride=1, padding=1)   # [B,8,H,W]
        self.conv1 = LightConvBnReLU6(8, 8, kernel_size=3, stride=1, padding=1)    # [B,8,H,W]

        self.conv2 = LightConvBnReLU6(8, 16, kernel_size=5, stride=2, padding=2)   # [B,16,H/2,W/2]
        self.conv5 = LightConvBnReLU6(16, 32, kernel_size=5, stride=2, padding=2)  # [B,32,H/4,W/4]
        self.conv8 = LightConvBnReLU6(32, 64, kernel_size=5, stride=2, padding=2)  # [B,64,H/8,W/8]

        # 多尺度融合层，保持输出特征尺寸不变（1x1卷积）
        self.output1 = nn.Conv2d(64, 64, kernel_size=1, bias=False)  # 用于 Stage 3（深层）
        self.inner1 = nn.Conv2d(32, 64, kernel_size=1, bias=True)      # 用于 Stage 2（中间），将 x2 从 32->64
        self.inner2 = nn.Conv2d(16, 64, kernel_size=1, bias=True)      # 用于 Stage 1（浅层），将 x1 从 16->64
        self.output2 = nn.Conv2d(64, 32, kernel_size=1, bias=False)     # Stage 2 输出，64->32
        self.output3 = nn.Conv2d(64, 16, kernel_size=1, bias=False)     # Stage 1 输出，64->16

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        output_feature = {}
        # Stage 1 (浅层信息)：利用 conv0 和 conv1
        x0 = self.conv1(self.conv0(x))    # [B,8,H,W]
        # Stage 2：下采样到 1/2 尺寸
        x1 = self.conv2(x0)               # [B,16,H/2,W/2] -> 作为浅层特征
        # Stage 3：下采样到 1/4 尺寸
        x2 = self.conv5(x1)               # [B,32,H/4,W/4] -> 作为中间特征
        # Stage 4：下采样到 1/8 尺寸
        x3 = self.conv8(x2)               # [B,64,H/8,W/8] -> 作为深层特征

        # Deep branch: Stage 3 输出
        output_feature[3] = self.output1(x3)  # [B,64,H/8,W/8]

        # Stage 2融合：将深层特征上采样到与中间特征一致，再与中间特征融合
        deep_up = F.interpolate(x3, scale_factor=2.0, mode="bilinear", align_corners=False)  # [B,64,H/4,W/4]
        fused_intermediate = deep_up + self.inner1(x2)  # inner1(x2): [B,64,H/4,W/4]
        output_feature[2] = self.output2(fused_intermediate)  # [B,32,H/4,W/4]

        # Stage 1融合：将 Stage 2 融合结果上采样到与浅层特征一致，再融合
        stage2_up = F.interpolate(fused_intermediate, scale_factor=2.0, mode="bilinear", align_corners=False)  # [B,64,H/2,W/2]
        fused_shallow = stage2_up + self.inner2(x1)  # inner2(x1): [B,64,H/2,W/2]
        output_feature[1] = self.output3(fused_shallow)  # [B,16,H/2,W/2]

        return output_feature
    
def apply_pruning(module, amount=0.3):

    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            prune.l1_unstructured(child, name="weight", amount=amount)
        else:
            apply_pruning(child, amount=amount)
