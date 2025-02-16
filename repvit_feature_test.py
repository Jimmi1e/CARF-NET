import torch
import torch.nn as nn

# Adjust these imports to match your actual file locations/names
from models.net import FeatureNet         # Original CNN FeatureNet
from models.repvit_feature import RepViTNet  # Your RepViT-based feature extractor

def compare_feature_nets_repvit():
    # Create a dummy input of shape [B, 3, H, W] with H=W=512
    dummy_input = torch.randn(1, 3, 512, 512)

    # Instantiate and evaluate the original CNN FeatureNet
    cnet = FeatureNet()
    cnet.eval()
    with torch.no_grad():
        cnet_outputs = cnet(dummy_input)

    # Instantiate and evaluate the RepViT-based feature extractor
    # If you have a checkpoint, you can pass it here, e.g. ckpt_path="./checkpoints/repvit_m1_5_distill_450e.pth"
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
