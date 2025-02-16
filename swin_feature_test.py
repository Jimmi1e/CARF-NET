
import torch
import torch.nn as nn

from models.net import FeatureNet
from models.swin_feature import SwinFeatureNet

def compare_feature_nets():
    dummy_input = torch.randn(1, 3, 512, 512)
    cnet = FeatureNet()
    cnet.eval()
    with torch.no_grad():
        cnet_outputs = cnet(dummy_input)
    
    snet = SwinFeatureNet(img_size=512, window_size=8)
    snet.eval()
    with torch.no_grad():
        snet_outputs = snet(dummy_input)
    
    print("=== CNN FeatureNet 输出形状 ===")
    for stage, feat in cnet_outputs.items():
        print(f"Stage {stage}: {feat.shape}")
    
    print("\n=== SwinFeatureNet 输出形状 ===")
    for stage, feat in snet_outputs.items():
        print(f"Stage {stage}: {feat.shape}")
    

    for stage in [1, 2, 3]:
        assert cnet_outputs[stage].shape == snet_outputs[stage].shape, \
            f"Stage {stage} 形状不匹配: CNN版 {cnet_outputs[stage].shape}, Swin版 {snet_outputs[stage].shape}"
    
    print("\nShape Match Pass：两者在 Stage 1/2/3 的输出形状一致！")

if __name__ == "__main__":
    compare_feature_nets()
