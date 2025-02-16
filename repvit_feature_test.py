import torch
import torch.nn as nn

from models.net import FeatureNet
# from models.repvit_feature import RepViTNet
from models.repvit_feature11 import RepViTNet
def compare_feature_nets_repvit():
    dummy_input = torch.randn(1, 3, 512, 512)

    cnet = FeatureNet()
    cnet.eval()
    with torch.no_grad():
        cnet_outputs = cnet(dummy_input)

    # rnet = RepViTNet11()
    rnet = RepViTNet()
    rnet.eval()
    with torch.no_grad():
        rnet_outputs = rnet(dummy_input)

    # Print the shapes from the original CNN FeatureNet
    print("=== CNN FeatureNet Output Shapes ===")
    for stage, feat in cnet_outputs.items():
        print(f"Stage {stage}: {feat.shape}")

    # Print the shapes from the RepViT-based FeatureNet
    print("\n=== RepViTFeatureNet Output Shapes ===")
    for stage, feat in rnet_outputs.items():
        print(f"Stage {stage}: {feat.shape}")

    # Compare shapes at stages 1, 2, and 3
    for stage in [1, 2, 3]:
        assert cnet_outputs[stage].shape == rnet_outputs[stage].shape, \
            f"Stage {stage} shape mismatch: CNN version {cnet_outputs[stage].shape}, RepViT version {rnet_outputs[stage].shape}"

    print("\nShape Match Pass: both FeatureNet and RepViTFeatureNet match at Stage 1/2/3!")

if __name__ == "__main__":
    compare_feature_nets_repvit()
