# import torch
# import torch.nn as nn

# from models.net_ori import FeatureNet
# from models.net_new import FeatureNet
# # from models.repvit_feature import RepViTNet
# #from models.repvit_feature11 import RepViTNet
# from models.liteFeatureNet import LightFeatureNet
# def compare_feature_nets_repvit():
#     dummy_input = torch.randn(1, 3, 512, 512)

#     cnet = FeatureNet()
#     cnet.eval()
#     with torch.no_grad():
#         cnet_outputs = cnet(dummy_input)

#     # rnet = RepViTNet11()
#     rnet = FeatureNet()
#     rnet.eval()
#     with torch.no_grad():
#         rnet_outputs = rnet(dummy_input)

#     # Print the shapes from the original CNN FeatureNet
#     print("=== CNN FeatureNet Output Shapes ===")
#     for stage, feat in cnet_outputs.items():
#         print(f"Stage {stage}: {feat.shape}")

#     # Print the shapes from the RepViT-based FeatureNet
#     print("\n=== RepViTFeatureNet Output Shapes ===")
#     for stage, feat in rnet_outputs.items():
#         print(f"Stage {stage}: {feat.shape}")

#     # Compare shapes at stages 1, 2, and 3
#     for stage in [1, 2, 3]:
#         assert cnet_outputs[stage].shape == rnet_outputs[stage].shape, \
#             f"Stage {stage} shape mismatch: CNN version {cnet_outputs[stage].shape}, RepViT version {rnet_outputs[stage].shape}"

#     print("\nShape Match Pass: both FeatureNet and RepViTFeatureNet match at Stage 1/2/3!")

# if __name__ == "__main__":
#     compare_feature_nets_repvit()
import torch

# 注意：这里导入时要确保两个 FeatureNet 模块路径正确，
# 如果两个模块类名都叫 FeatureNet，可以用不同的名字导入
from models.net_ori import FeatureNet as OriginalFeatureNet
from models.net_new import FeatureNet as NewFeatureNet

def compare_feature_nets():
    dummy_input = torch.randn(1, 3, 512, 512)

    # 原始 FeatureNet
    ori_net = OriginalFeatureNet()
    ori_net.eval()
    with torch.no_grad():
        ori_outputs = ori_net(dummy_input)

    # 新 FeatureNet
    new_net = NewFeatureNet()
    new_net.eval()
    with torch.no_grad():
        new_outputs = new_net(dummy_input)

    # 打印原始 FeatureNet 各阶段输出的形状
    print("=== Original FeatureNet Output Shapes ===")
    for stage, feat in ori_outputs.items():
        print(f"Stage {stage}: {feat.shape}")

    # 打印新 FeatureNet 各阶段输出的形状
    print("\n=== New FeatureNet Output Shapes ===")
    for stage, feat in new_outputs.items():
        print(f"Stage {stage}: {feat.shape}")

    # 对比 stages 1, 2, 3 的输出形状是否一致
    for stage in ["stage_1", "stage_2", "stage_3"]:
        assert ori_outputs[stage].shape == new_outputs[stage].shape, \
            f"Stage {stage} shape mismatch: Original {ori_outputs[stage].shape}, New {new_outputs[stage].shape}"
    print("\nShape Match Pass: both networks produce matching shapes at Stage_1, Stage_2 and Stage_3!")

if __name__ == "__main__":
    compare_feature_nets()
