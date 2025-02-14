import torch
import torch.nn as nn
import torch.nn.functional as F
from swin_transformer_v2 import PatchEmbed, BasicLayer, PatchMerging

class SwinFeatureNet(nn.Module):
    """
    Swin Transformer as the feature extraction backbone for PatchMatchNet.
    """

    def __init__(self, img_size=512):
        super(SwinFeatureNet, self).__init__()
        # Swin Transformer config
        self.embed_dim = 96  #embedding dimension
        self.depths = [2, 2, 6]  # 3 stages to match FeatureNet's output levels 不是【2，2，6，2】
        self.num_heads = [3, 6, 12]  # Attention heads per stage 不是【3，6，12，24】

        # Patch embedding layer (splitting image into tokens)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=4, in_chans=3, embed_dim=self.embed_dim)

        patches_resolution = self.patch_embed.patches_resolution
        self.num_layers = len(self.depths)

        # Swin Transformer hierarchical layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(self.embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=7,
                mlp_ratio=4.,
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=0.1,
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
            )
            self.layers.append(layer)


        # Extra 1x1 convolutions to match FeatureNet's output channel sizes
        self.output1 = nn.Conv2d(self.embed_dim * 4, 64, 1, bias=False)  # Swin Stage 3 → 64 channels
        self.inner1 = nn.Conv2d(self.embed_dim * 2, 64, 1, bias=True)
        self.inner2 = nn.Conv2d(self.embed_dim, 64, 1, bias=True)
        self.output2 = nn.Conv2d(self.embed_dim * 2, 32, 1, bias=False)  # Swin Stage 2 → 32 channels
        self.output3 = nn.Conv2d(self.embed_dim, 16, 1, bias=False)  # Swin Stage 1 → 16 channels

    def forward(self, x: torch.Tensor):
        """
        Input: x: Image tensor [B, 3, H, W]
        Output: output_feature: Dictionary with features from 3 stages
        """
        #patch embeddings
        x = self.patch_embed(x)  # Shape: [B, N, C]
        features = {}

        # interation all stage
        for i, layer in enumerate(self.layers):
            x = layer(x)  #output：[B, N, C]，C will be increased with stage
            B, L, C = x.shape
            H = self.patch_embed.patches_resolution[0] // (2 ** i)
            W = self.patch_embed.patches_resolution[1] // (2 ** i)
            x_reshaped = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
            features[i] = x_reshaped

        # Stage 3 feature
        f3 = self.output1(features[2])  # [B, 64, H/4, W/4]

        # Stage 2 feature fusion：f3 上采样到 Stage 2 的分辨率，fusion with inner1(features[1])
        f2_inner = self.inner1(features[1])  # [B, 64, H/2, W/2]
        f3_up = F.interpolate(f3, scale_factor=2.0, mode='bilinear', align_corners=False)  # [B, 64, H/2, W/2]
        fused_f2 = f3_up + f2_inner
        f2_out = self.output2(fused_f2)  # [B, 32, H/2, W/2]

        # Stage 1 feature fusion：将 fused_f2 上采样到 Stage 1 的分辨率，fusion with inner2(features[0]) 
        f1_inner = self.inner2(features[0])  # [B, 64, H, W]
        fused_f1 = F.interpolate(fused_f2, scale_factor=2.0, mode='bilinear', align_corners=False) + f1_inner  # [B, 64, H, W]
        f1_out = self.output3(fused_f1)  # [B, 16, H, W]

        output_feature = {3: f3, 2: f2_out, 1: f1_out}
        return output_feature
