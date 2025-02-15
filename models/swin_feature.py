import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_transformer_v2 import PatchEmbed, BasicLayer, PatchMerging

class SwinFeatureNet(nn.Module):
    """
    Swin Transformer as the feature extraction backbone for PatchMatchNet.
    """
    def __init__(self, img_size=512, window_size=8):
        super(SwinFeatureNet, self).__init__()
        self.embed_dim = 96
        self.depths = [2, 2, 6]
        self.num_heads = [3, 6, 12]

        # Patch embedding：512×512 -----divede--->256×256 patch，out put channel is 96
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=2, in_chans=3, embed_dim=self.embed_dim) #original patch size is 4
        patches_resolution = self.patch_embed.patches_resolution  # (256, 256)
        self.num_layers = len(self.depths)

        # Swin Transformer Layer
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(self.embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=4.,
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=0.1,
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
            )
            self.layers.append(layer)

        # features[0]（Layer 0）：After PatchMerging = embed_dim * 2 = 192
        # features[1]（Layer 1）：After PatchMerging = embed_dim * 4 = 384
        # features[2]（Layer 2）：channel = embed_dim * 4 = 384 without downsampling
        self.output1 = nn.Conv2d(self.embed_dim * 4, 64, 1, bias=False)   # 384 -> 64， Stage 3
        self.inner1 = nn.Conv2d(self.embed_dim * 4, 64, 1, bias=True)        # 384 -> 64， Stage 2
        self.inner2 = nn.Conv2d(self.embed_dim * 2, 64, 1, bias=True)        # 192 -> 64，Stage 1
        # After fusion
        self.output2 = nn.Conv2d(64, 32, 1, bias=False)  # Stage 2 out：64 -> 32
        self.output3 = nn.Conv2d(64, 16, 1, bias=False)  # Stage 1 out：64 -> 16

    def forward(self, x: torch.Tensor):
        # Patch embedding
        x = self.patch_embed(x)  # [B, 256*256, 96]
        features = {}

        for i, layer in enumerate(self.layers):
            x = layer(x)  # [B, L, C]
            B, L, C = x.shape
            # layer0：The original patch_embed resolution is 128×128 ， after PatchMerging ---> 128/2=64
            # layer1：downsampling，64/2=32
            # layer2：keep 32
            if i == 0:
                H, W = self.patch_embed.patches_resolution[0] // 2, self.patch_embed.patches_resolution[1] // 2  # 128//2 = 64
            else:
                H, W = self.patch_embed.patches_resolution[0] // 4, self.patch_embed.patches_resolution[1] // 4  # 128//4 = 32
            x_reshaped = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            features[i] = x_reshaped

        # features[0]: [B, 192, 64, 64]
        # features[1]: [B, 384, 32, 32]
        # features[2]: [B, 384, 32, 32]

        # Stage 3：直接对 features[2] 映射
        f3 = self.output1(features[2])  # [B, 64, 32, 32]

        # Stage 2：对 features[1] 先映射再上采样到 64×64
        f2_inner = self.inner1(features[1])  # [B, 64, 32, 32]
        f2_inner = F.interpolate(f2_inner, scale_factor=2.0, mode='bilinear', align_corners=False)  # [B, 64, 64, 64]
        # 同时将 f3 上采样到 64×64
        f3_up = F.interpolate(f3, scale_factor=2.0, mode='bilinear', align_corners=False)  # [B, 64, 64, 64]
        fused_f2 = f3_up + f2_inner  # [B, 64, 64, 64]
        f2_out = self.output2(fused_f2)  # [B, 32, 64, 64]

        # Stage 1：对 features[0] 先映射再上采样到 128×128
        f1_inner = self.inner2(features[0])  # [B, 64, 64, 64]
        f1_inner = F.interpolate(f1_inner, scale_factor=2.0, mode='bilinear', align_corners=False)  # [B, 64, 128, 128]
        # 将 Stage 2 融合结果上采样到 128×128，与 f1_inner 融合
        fused_f1 = F.interpolate(fused_f2, scale_factor=2.0, mode='bilinear', align_corners=False) + f1_inner  # [B, 64, 128, 128]
        f1_out = self.output3(fused_f1)  # [B, 16, 128, 128]

        output_feature = {3: f3, 2: f2_out, 1: f1_out}
        return output_feature
