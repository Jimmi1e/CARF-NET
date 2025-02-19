import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from .repvit import repvit_m0_9

class RepViTNet09(nn.Module):
    """
    Feature extraction network using the RepViT backbone.
    """
    def __init__(self, ckpt_path=None):
        super(RepViTNet09, self).__init__()
        self.backbone = repvit_m0_9(pretrained=False)

        self.output1 = nn.Conv2d(384, 192, kernel_size=1, bias=False)
        self.inner1  = nn.Conv2d(192, 192, kernel_size=1, bias=True)
        self.inner2  = nn.Conv2d(96, 48, kernel_size=1, bias=True)
        self.inner3  = nn.Conv2d(96, 48, kernel_size=1, bias=True)
        self.output2 = nn.Conv2d(192, 96, kernel_size=1, bias=False)
        self.output3 = nn.Conv2d(48, 32, kernel_size=1, bias=False)
        
        if ckpt_path is not None:
            self.load_ckpt_weights(ckpt_path)
            
    def load_ckpt_weights(self, ckpt_path):
        """Load pre-trained weights from the checkpoint."""
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
        shallow_feature = features[7] #shallow 
        intermediate_feature = features[23]#intermediate
        deep_feature = features[-1]#dedep 
        
        # Stage 3 (Deep):
        stage3 = self.output1(deep_feature)# [B, 64, 16, 16]
        stage3_upsampled = F.interpolate(stage3, scale_factor=4.0, mode="bilinear", align_corners=False)  # [B, 64, 64, 64]
        
        output_feature[3] = stage3_upsampled
        print( output_feature[3].shape)
        # Stage 2 (Intermediate):
        deep_up_for_stage2 = F.interpolate(stage3, scale_factor=2.0, mode="bilinear", align_corners=False)  # [B, 64, 32, 32]
        projected_intermediate = self.inner1(intermediate_feature)  # [B, 64, 32, 32]
        intra_feat_stage2 = deep_up_for_stage2 + projected_intermediate  # [B, 64, 32, 32]
        intra_feat_stage2_upsampled = F.interpolate(intra_feat_stage2, scale_factor=4.0, mode="bilinear", align_corners=False)  # [B, 64, 128, 128]
        
        output_feature[2] = self.output2(intra_feat_stage2_upsampled)  # [B, 32, 128, 128]
        print(output_feature[2].shape)
        # Stage 1 (Shallow):
        shallow_proj = self.inner2(shallow_feature)  # [B, 64, 128, 128]
        
        stage2_for_stage1 = F.interpolate(output_feature[2], scale_factor=2.0, mode="bilinear", align_corners=False)  # [B, 64, 256, 256]
        stage2_for_stage1=self.inner3(stage2_for_stage1)
        shallow_proj_upsampled=F.interpolate(shallow_proj, scale_factor=4.0, mode="bilinear", align_corners=False)
        
        intra_feat_stage1 = stage2_for_stage1 +shallow_proj_upsampled   # [B, 64, 256, 256]
        output_feature[1] = self.output3(intra_feat_stage1)  # [B, 16, 256, 256]
        print(output_feature[1].shape)
        return output_feature
