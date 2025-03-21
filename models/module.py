"""
Implementation of Pytorch layer primitives, such as Conv+BN+ReLU, differentiable warping layers,
and depth regression based upon expectation of an input probability distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordAtt(nn.Module):
     def __init__(self, inp, oup, groups=32):
         super(CoordAtt, self).__init__()
         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
 
         mip = max(8, inp // groups)
 
         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
         self.bn1 = nn.BatchNorm2d(mip)
         self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
         self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
         self.relu = nn.ReLU(inplace=True)  # 替换 h_swish()
 
     def forward(self, x):
         identity = x
         n, c, h, w = x.size()
         x_h = self.pool_h(x)
         x_w = self.pool_w(x).permute(0, 1, 3, 2)
 
         y = torch.cat([x_h, x_w], dim=2)
         y = self.conv1(y)
         y = self.bn1(y)
         y = self.relu(y)
         x_h, x_w = torch.split(y, [h, w], dim=2)
         x_w = x_w.permute(0, 1, 3, 2)
 
         x_h = self.conv2(x_h).sigmoid()
         x_w = self.conv3(x_w).sigmoid()
         x_h = x_h.expand(-1, -1, h, w)
         x_w = x_w.expand(-1, -1, h, w)
 
         y = identity * x_w * x_h
 
         return y
class ConvBnReLU(nn.Module):
    """Implements 2d Convolution + batch normalization + ReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        pad: int = 1,
        dilation: int = 1,
    ) -> None:
        """initialization method for convolution2D + batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBnReLU3D(nn.Module):
    """Implements of 3d convolution + batch normalization + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        pad: int = 1,
        dilation: int = 1,
    ) -> None:
        """initialization method for convolution3D + batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return F.relu(self.bn(self.conv(x)), inplace=True)


class DepthAxialAttention3D(nn.Module):
    """DepthAxialAttention module，enhance z direction feature(depth information)"""
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels//reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels//reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.depth_conv = nn.Conv3d(
            in_channels, in_channels, 
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=in_channels
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(in_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x_ca = x * ca     
        z_feat = self.depth_conv(x_ca)
        
        sa = self.spatial_attention(z_feat)
        out = x_ca * sa
        return out + x  # residual connection

class ConvBNReLU3D_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, stride=1, padding=1, dilation: int = 1,
                 attention_type='axial'):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 
                     kernel_size, stride, padding,dilation, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        if attention_type == 'axial':
            self.attention = DepthAxialAttention3D(out_channels)
        elif attention_type == 'cbam':
            self.attention = CBAM3D(out_channels)
        elif attention_type == 'se':
            self.attention = scSE3D(out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        return self.attention(x)

class CBAM3D(nn.Module):
    """
    3D CBAM module
    This module computes both channel and spatial attention using 
    adaptive average & max pooling and 1x1x1 convolutions.
    
    Args:
        in_channels: Number of input channels.
        reduction_ratio: Reduction ratio for channel attention.
        spatial_kernel: Kernel size for the spatial attention convolution.
    """
    def __init__(self, in_channels, reduction_ratio=8, spatial_kernel=7):
        super(CBAM3D, self).__init__()
        # Channel Att
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        #Spatial Att
        self.conv_spatial = nn.Conv3d(2, 1, kernel_size=spatial_kernel, 
                                      padding=(spatial_kernel - 1) // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        #x:[B, C, D, H, W]
        #Channel Attention
        avg_out = self.mlp(self.avg_pool(x))  # [B, C, 1, 1, 1]
        max_out = self.mlp(self.max_pool(x))  # [B, C, 1, 1, 1]
        channel_att = self.sigmoid_channel(avg_out + max_out)  # [B, C, 1, 1, 1]
        x_channel = x * channel_att

        # Spatial Attention
        avg_out_spatial = torch.mean(x_channel, dim=1, keepdim=True)  # [B,1,D,H,W]
        max_out_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)  # [B,1,D,H,W]
        spatial_cat = torch.cat([avg_out_spatial, max_out_spatial], dim=1)  # [B,2,D,H,W]
        spatial_att = self.sigmoid_spatial(self.conv_spatial(spatial_cat))  # [B,1,D,H,W]
        out = x_channel * spatial_att
        return out

# class HybridAttentionNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         #
#         self.encoder1 = ConvBNReLU3D_Attention(3, 64, attention_type='axial')
#         # Deep layre CBAM
#         self.encoder2 = ConvBNReLU3D_Attention(64, 128, attention_type='cbam')
#         #Mildde layer dont use any attention module
#         self.bottleneck = ConvBNReLU3D_Attention(128, 256)

class sSE3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = nn.Conv3d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, U):
        # U: [bs, c, d, h, w] -> q: [bs, 1, d, h, w]
        q = self.conv1x1(U)
        q = self.sigmoid(q)
        return U * q

class cSE3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # using 3D adaptive pooling ，squeeze spatial volume to 1×1×1
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.conv_squeeze = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.conv_excitation = nn.Conv3d(in_channels // 2, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, U):
        # [bs, c, d, h, w] to [bs, c, 1, 1, 1]
        z = self.avgpool(U)
        z = self.conv_squeeze(z)
        z = self.conv_excitation(z)
        z = self.sigmoid(z)
        return U * z.expand_as(U)

class scSE3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE3D(in_channels)
        self.sSE = sSE3D(in_channels)

    def forward(self, U):
        U_cse = self.cSE(U)
        U_sse = self.sSE(U)
        return U_cse + U_sse

class ConvBnReLU1D(nn.Module):
    """Implements 1d Convolution + batch normalization + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        pad: int = 1,
        dilation: int = 1,
    ) -> None:
        """initialization method for convolution1D + batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super(ConvBnReLU1D, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    """Implements of 2d convolution + batch normalization."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, pad: int = 1
    ) -> None:
        """initialization method for convolution2D + batch normalization + ReLU module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
        """
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return self.bn(self.conv(x))


def differentiable_warping(
    src_fea: torch.Tensor, src_proj: torch.Tensor, ref_proj: torch.Tensor, depth_samples: torch.Tensor
):
    """Differentiable homography-based warping, implemented in Pytorch.

    Args:
        src_fea: [B, C, Hin, Win] source features, for each source view in batch
        src_proj: [B, 4, 4] source camera projection matrix, for each source view in batch
        ref_proj: [B, 4, 4] reference camera projection matrix, for each ref view in batch
        depth_samples: [B, Ndepth, Hout, Wout] virtual depth layers
    Returns:
        warped_src_fea: [B, C, Ndepth, Hout, Wout] features on depths after perspective transformation
    """

    batch, num_depth, height, width = depth_samples.shape
    channels = src_fea.shape[1]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid(
            [
                torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                torch.arange(0, width, dtype=torch.float32, device=src_fea.device),
            ]
        )
        y, x = y.contiguous().view(height * width), x.contiguous().view(height * width)
        xyz = torch.unsqueeze(torch.stack((x, y, torch.ones_like(x))), 0).repeat(batch, 1, 1)  # [B, 3, H*W]

        rot_depth_xyz = torch.matmul(rot, xyz).unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_samples.view(
            batch, 1, num_depth, height * width
        )  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # avoid negative depth
        negative_depth_mask = proj_xyz[:, 2:] <= 1e-3
        proj_xyz[:, 0:1][negative_depth_mask] = float(width)
        proj_xyz[:, 1:2][negative_depth_mask] = float(height)
        proj_xyz[:, 2:3][negative_depth_mask] = 1.0
        grid = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = grid[:, 0, :, :] / ((width - 1) / 2) - 1  # [B, Ndepth, H*W]
        proj_y_normalized = grid[:, 1, :, :] / ((height - 1) / 2) - 1
        grid = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]

    return F.grid_sample(
        src_fea,
        grid.view(batch, num_depth * height, width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(batch, channels, num_depth, height, width)


def depth_regression(p: torch.Tensor, depth_values: torch.Tensor) -> torch.Tensor:
    """Implements per-pixel depth regression based upon a probability distribution per-pixel.

    The regressed depth value D(p) at pixel p is found as the expectation w.r.t. P of the hypotheses.

    Args:
        p: probability volume [B, D, H, W]
        depth_values: discrete depth values [B, D]
    Returns:
        result depth: expected value, soft argmin [B, 1, H, W]
    """

    return torch.sum(p * depth_values.view(depth_values.shape[0], 1, 1), dim=1).unsqueeze(1)


def is_empty(x: torch.Tensor) -> bool:
    return x.numel() == 0

