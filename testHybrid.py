import torch
from models.net_new import HybridPatchMatchNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模拟输入数据
B = 2         # 批次大小
N = 3         # 视角数（参考视角 + 2个源视角）
C = 3         # 通道数
H = 128       # 图像高度
W = 160       # 图像宽度

# 随机生成多视角图像，形状 [B, N, C, H, W]
images = torch.randn(B, N, C, H, W, device=device)

# 构造内参矩阵，形状 [B, N, 3, 3]
intrinsics = torch.zeros(B, N, 3, 3, device=device)
for b in range(B):
    for n in range(N):
        intrinsics[b, n] = torch.tensor([[100.0, 0, W/2],
                                          [0, 100.0, H/2],
                                          [0, 0, 1]], device=device)

# 构造外参矩阵（单位矩阵作为示例），形状 [B, N, 4, 4]
extrinsics = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)

# 虚拟深度范围，形状 [B, 1]
depth_min = torch.full((B, 1), 1.0, device=device)
depth_max = torch.full((B, 1), 10.0, device=device)

# 定义 HybridPatchMatchNet 参数（示例参数）
patchmatch_interval_scale = [0.01, 0.01, 0.01]
propagation_range = [2, 2, 2]
patchmatch_iteration = [3, 3, 3]
patchmatch_num_sample = [48, 32, 16]
propagate_neighbors = [16, 8, 8]
evaluate_neighbors = [9, 9, 9]

# 创建模型
model = HybridPatchMatchNet(patchmatch_interval_scale, propagation_range,
                            patchmatch_iteration, patchmatch_num_sample,
                            propagate_neighbors, evaluate_neighbors,
                            featureNet='FeatureNet', Attention_Selection='None',
                            image_size=(H, W)).to(device)
model.eval()

with torch.no_grad():
    depth_final, photometric_conf, depth_patchmatch = model(images, intrinsics, extrinsics, depth_min, depth_max)

print("Final depth shape:", depth_final.shape)
print("Photometric confidence shape:", photometric_conf.shape)
for stage, d_list in depth_patchmatch.items():
    print(f"Stage {stage} has {len(d_list)} depth map(s) with shape {d_list[0].shape}")
