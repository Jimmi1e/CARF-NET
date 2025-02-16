import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from .repvit import repvit_m1_5

class RepViTNet(nn.Module):
    """
    Feature extraction network using the RepViT backbone.
    """
    def __init__(self, ckpt_path=None):
        super(RepViTNet, self).__init__()
        self.backbone = repvit_m1_5(pretrained=False)

        self.output1 = nn.Conv2d(512, 64, kernel_size=1, bias=False)
        self.inner1  = nn.Conv2d(512, 64, kernel_size=1, bias=True)
        self.inner2  = nn.Conv2d(512, 64, kernel_size=1, bias=True)
        self.output2 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.output3 = nn.Conv2d(64, 16, kernel_size=1, bias=False)
        
        if ckpt_path is not None:
            self.load_ckpt_weights(ckpt_path)
            
    def load_ckpt_weights(self, ckpt_path):
        """
        Load pre-trained weights from the checkpoint.
        If the checkpoint contains a 'state_dict' key, use its content.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        self.backbone.load_state_dict(ckpt, strict=False)
        
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        features: List[torch.Tensor] = []
        for f in self.backbone.features:
            x = f(x)
            features.append(x)
            
        output_feature: Dict[int, torch.Tensor] = {}
        conv10 = features[-1]  # deep feature
        conv7  = features[-3]  # intermediate feature
        conv4  = features[-5]  # shallow feature
        
        # Stage 3: 
        stage3 = self.output1(conv10)
        stage3_upsampled = F.interpolate(stage3, scale_factor=4.0, mode="bilinear", align_corners=False)
        output_feature[3] = stage3_upsampled
        
        # Stage 2:
        stage3_upsampled_for_stage2 = F.interpolate(stage3, scale_factor=8.0, mode="bilinear", align_corners=False)
        projected_conv7 = self.inner1(conv7)
        projected_conv7_upsampled = F.interpolate(projected_conv7, scale_factor=8.0, mode="bilinear", align_corners=False)
        # Fuse the two branches.
        intra_feat_stage2 = stage3_upsampled_for_stage2 + projected_conv7_upsampled
        output_feature[2] = self.output2(intra_feat_stage2)
        
        # Stage 1:
        intra_feat_stage2_upsampled = F.interpolate(intra_feat_stage2, scale_factor=2.0, mode="bilinear", align_corners=False)

        projected_conv4 = self.inner2(conv4)
        projected_conv4_upsampled = F.interpolate(projected_conv4, scale_factor=16.0, mode="bilinear", align_corners=False)
        # Fuse the two branches.
        intra_feat_stage1 = intra_feat_stage2_upsampled + projected_conv4_upsampled
        output_feature[1] = self.output3(intra_feat_stage1)
        
        return output_feature
