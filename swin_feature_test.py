# import torch
# from swin_feature import SwinFeatureNet

# def test_swin_feature_net():
#     # random matrix [B, 3, H, W]
#     batch_size = 1
#     img_size = 512
#     dummy_input = torch.randn(batch_size, 3, img_size, img_size)

#     model = SwinFeatureNet(img_size=img_size)
#     model.eval()
    
#     with torch.no_grad():
#         outputs = model(dummy_input)
    
#     print("输出各阶段特征形状：")
#     for stage, feat in outputs.items():
#         print(f"Stage {stage}: {feat.shape}")
    
#     # 
#     # - PatchEmbed: 512×512 划分为 128×128 的 patch（512/4），
#     # - stage1（features[0]）：shape[B, 96, 128, 128]，
#     # - stage2（features[1]）：After down sampling，Sahpe [B, 192, 64, 64]，
#     # - stage3（features[2]）：Shape[B, 384, 32, 32]，
#     # Output
#     #   Stage 3 (f3): [B, 64, 32, 32]
#     #   Stage 2 (f2_out): [B, 32, 64, 64]
#     #   Stage 1 (f1_out): [B, 16, 128, 128]
#     expected_stage3_shape = (batch_size, 64, img_size // 16, img_size // 16)  # 512//16 = 32
#     expected_stage2_shape = (batch_size, 32, img_size // 8, img_size // 8)    # 512//8  = 64
#     expected_stage1_shape = (batch_size, 16, img_size // 4, img_size // 4)    # 512//4  = 128

#     assert outputs[3].shape == expected_stage3_shape, f"Stage 3 形状不匹配: 期望 {expected_stage3_shape}, 得到 {outputs[3].shape}"
#     assert outputs[2].shape == expected_stage2_shape, f"Stage 2 形状不匹配: 期望 {expected_stage2_shape}, 得到 {outputs[2].shape}"
#     assert outputs[1].shape == expected_stage1_shape, f"Stage 1 形状不匹配: 期望 {expected_stage1_shape}, 得到 {outputs[1].shape}"
    
#     print("所有输出形状测试均已通过！")

# if __name__ == '__main__':
#     test_swin_feature_net()
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
