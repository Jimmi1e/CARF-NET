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

        self.inner1 = nn.Conv2d(self.embed_dim * 2, 64, 1, bias=True)
        self.inner2 = nn.Conv2d(self.embed_dim, 64, 1, bias=True)
        # Extra 1x1 convolutions to match FeatureNet's output channel sizes
        self.output1 = nn.Conv2d(self.embed_dim * 4, 64, 1, bias=False)  # Swin Stage 3 → 64 channels
        self.output2 = nn.Conv2d(self.embed_dim * 2, 32, 1, bias=False)  # Swin Stage 2 → 32 channels
        self.output3 = nn.Conv2d(self.embed_dim, 16, 1, bias=False)  # Swin Stage 1 → 16 channels

    def forward(self, x: torch.Tensor):
        """
        Input:x: Image tensor [B, 3, H, W]
        Output: output_feature: Dictionary with features from 3 stages
        """
        output_feature = {}
        # Convert image into patch embeddings
        x = self.patch_embed(x)  # Shape: [B, N, C]

        for i, layer in enumerate(self.layers):
            x = layer(x)  # Process with Swin Transformer layers
            B, L, C = x.shape
            H = self.patch_embed.patches_resolution[0] // (2 ** i)
            W = self.patch_embed.patches_resolution[1] // (2 ** i)
            x_reshaped = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

            # Save feature maps at 3 different scales
            if i == 0:
                output_feature[1] = self.output3(x_reshaped) #64 channels
            elif i == 1:
                output_feature[2] = self.output2(x_reshaped)#32 channels
            elif i == 2:
                output_feature[3] = self.output1(x_reshaped)#16 channels

        return output_feature
