import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Dummy 实现部分 ---
def adjust_image_dims(images, intrinsics):
    # 此处假设输入尺寸已经符合要求，直接返回
    B, C, H, W = images[0].shape
    return images, intrinsics, H, W

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Dummy PatchMatch 模块，返回随机张量模拟输出
class PatchMatch(nn.Module):
    def __init__(self, propagation_out_range, patchmatch_iteration, patchmatch_num_sample,
                 patchmatch_interval_scale, num_feature, G, propagate_neighbors,
                 evaluate_neighbors, stage, Attention_Selection):
        super(PatchMatch, self).__init__()
        # 保存参数，方便调试（如果需要）
        self.stage = stage
        self._num_depth = patchmatch_num_sample  # 作为 dummy 值使用

    def forward(self, ref_feature, src_features, ref_proj, src_projs,
                depth_min, depth_max, depth, view_weights):
        B, C, H, W = ref_feature.shape
        # 模拟3次迭代，每次输出一个深度图
        depths = [torch.rand(B, 1, H, W, device=ref_feature.device) for _ in range(3)]
        # 模拟 score 张量，通道数取 _num_depth 的第一个元素（假设是 int）
        num_depth = self._num_depth if isinstance(self._num_depth, int) else self._num_depth[0]
        score = torch.rand(B, num_depth, H, W, device=ref_feature.device)
        view_weights = torch.rand(B, 1, H, W, device=ref_feature.device)
        return depths, score, view_weights

# Dummy depth_regression：简单返回 score 的 argmax（作为 float 类型）
def depth_regression(score, depth_values):
    # 这里简单返回每个像素 score 通道上最大的索引作为深度回归结果
    return torch.argmax(score, dim=1, keepdim=True).float()

# --- 修改后的网络实现 ---

class FeatureNet(nn.Module):
    """Feature Extraction Network: 仅提取高分辨率 (H/2) 和低分辨率 (H/8) 特征"""

    def __init__(self):
        super(FeatureNet, self).__init__()

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)
        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.conv7 = ConvBnReLU(32, 32, 3, 1, 1)
        self.conv8 = ConvBnReLU(32, 64, 5, 2, 2)
        self.conv9 = ConvBnReLU(64, 64, 3, 1, 1)
        self.conv10 = ConvBnReLU(64, 64, 3, 1, 1)

        # 低分辨率输出：H/8
        self.output1 = nn.Conv2d(64, 64, 1, bias=False)
        # 中间层（用于高分辨率特征融合）
        self.inner1 = nn.Conv2d(32, 64, 1, bias=True)
        self.inner2 = nn.Conv2d(16, 64, 1, bias=True)
        # 高分辨率特征输出：H/2
        self.output3 = nn.Conv2d(64, 16, 1, bias=False)

    def forward(self, x: torch.Tensor) -> dict:
        conv1 = self.conv1(self.conv0(x))
        conv4 = self.conv4(self.conv3(self.conv2(conv1)))
        conv7 = self.conv7(self.conv6(self.conv5(conv4)))
        conv10 = self.conv10(self.conv9(self.conv8(conv7)))
        
        # 低分辨率特征（stage3）
        out_low = self.output1(conv10)
        
        # 高分辨率特征：对 conv10 先上采样，再与 conv7 和 conv4 融合
        intra_feat = F.interpolate(conv10, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner1(conv7)
        intra_feat = F.interpolate(intra_feat, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner2(conv4)
        out_high = self.output3(intra_feat)
        
        return {3: out_low, 1: out_high}


class Refinement(nn.Module):
    """Depth map refinement network"""

    def __init__(self):
        super(Refinement, self).__init__()

        # 输入图像分支
        self.conv0 = ConvBnReLU(in_channels=3, out_channels=8)
        # 深度图分支（下采样一半）
        self.conv1 = ConvBnReLU(in_channels=1, out_channels=8)
        self.conv2 = ConvBnReLU(in_channels=8, out_channels=8)
        self.deconv = nn.ConvTranspose2d(
            in_channels=8, out_channels=8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
        )

        self.bn = nn.BatchNorm2d(8)
        self.conv3 = ConvBnReLU(in_channels=16, out_channels=8)
        self.res = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1, bias=False)

    def forward(self, img: torch.Tensor, depth_0: torch.Tensor, depth_min: torch.Tensor, depth_max: torch.Tensor) -> torch.Tensor:
        batch_size = depth_min.size()[0]
        depth = (depth_0 - depth_min.view(batch_size, 1, 1, 1)) / (depth_max - depth_min).view(batch_size, 1, 1, 1)

        conv0 = self.conv0(img)
        deconv = F.relu(self.bn(self.deconv(self.conv2(self.conv1(depth)))), inplace=True)
        res = self.res(self.conv3(torch.cat((deconv, conv0), dim=1)))
        depth = F.interpolate(depth, scale_factor=2.0, mode="nearest") + res
        return depth * (depth_max - depth_min).view(batch_size, 1, 1, 1) + depth_min.view(batch_size, 1, 1, 1)


class PatchmatchNet(nn.Module):
    """修改后的 PatchmatchNet：仅使用 stage3 和 stage1 的 patchmatch 模块"""

    def __init__(
        self,
        patchmatch_interval_scale: list,
        propagation_range: list,
        patchmatch_iteration: list,
        patchmatch_num_sample: list,
        propagate_neighbors: list,
        evaluate_neighbors: list,
        featureNet='FeatureNet',
        Attention_Selection='None',
        image_size=(512,512)
    ) -> None:
        super(PatchmatchNet, self).__init__()

        # 只使用 FeatureNet
        if featureNet == 'FeatureNet':
            self.feature = FeatureNet()
        # 此处其他分支省略
        self.patchmatch_num_sample = patchmatch_num_sample

        num_features = [16, 32, 64]  # 各阶段特征通道数

        self.propagate_neighbors = propagate_neighbors
        self.evaluate_neighbors = evaluate_neighbors
        self.G = [4, 8, 8]

        # 创建仅有的两个 PatchMatch 模块：
        # stage1 对应高分辨率（H/2）
        self.patchmatch_1 = PatchMatch(
            propagation_out_range=propagation_range[0],
            patchmatch_iteration=patchmatch_iteration[0],
            patchmatch_num_sample=patchmatch_num_sample[0],
            patchmatch_interval_scale=patchmatch_interval_scale[0],
            num_feature=num_features[0],
            G=self.G[0],
            propagate_neighbors=self.propagate_neighbors[0],
            evaluate_neighbors=evaluate_neighbors[0],
            stage=1,
            Attention_Selection=Attention_Selection
        )
        # stage3 对应低分辨率（H/8）
        self.patchmatch_3 = PatchMatch(
            propagation_out_range=propagation_range[2],
            patchmatch_iteration=patchmatch_iteration[2],
            patchmatch_num_sample=patchmatch_num_sample[2],
            patchmatch_interval_scale=patchmatch_interval_scale[2],
            num_feature=num_features[2],
            G=self.G[2],
            propagate_neighbors=self.propagate_neighbors[2],
            evaluate_neighbors=evaluate_neighbors[2],
            stage=3,
            Attention_Selection=Attention_Selection
        )

        self.upsample_net = Refinement()

    def forward(self, images: list, intrinsics: torch.Tensor, extrinsics: torch.Tensor,
                depth_min: torch.Tensor, depth_max: torch.Tensor):
        assert len(images) == intrinsics.size()[1], "不同数量的图像与内参矩阵"
        assert len(images) == extrinsics.size()[1], "不同数量的图像与外参矩阵"
        images, intrinsics, orig_height, orig_width = adjust_image_dims(images, intrinsics)
        ref_image = images[0]
        _, _, ref_height, ref_width = ref_image.size()

        # Step 1. 多尺度特征提取（仅返回 key 3 和 1）
        features = []
        for img in images:
            output_feature = self.feature(img)
            features.append(output_feature)
        del images
        ref_feature, src_features = features[0], features[1:]

        depth_min = depth_min.float()
        depth_max = depth_max.float()
        device = intrinsics.device

        depth_patchmatch = {}

        # Step 2. PatchMatch - 低分辨率 stage3 (H/8, scale=0.125)
        scale = 0.125
        src_features_l = [src_fea[3] for src_fea in src_features]
        intrinsics_l = intrinsics.clone()
        intrinsics_l[:, :, :2] *= scale
        proj = extrinsics.clone()
        proj[:, :, :3, :4] = torch.matmul(intrinsics_l, extrinsics[:, :, :3, :4])
        proj_l = torch.unbind(proj, 1)
        ref_proj, src_proj = proj_l[0], proj_l[1:]
        
        depth = torch.empty(0, device=device)
        view_weights = torch.empty(0, device=device)
        depths, score, view_weights = self.patchmatch_3(
            ref_feature=ref_feature[3],
            src_features=src_features_l,
            ref_proj=ref_proj,
            src_projs=src_proj,
            depth_min=depth_min,
            depth_max=depth_max,
            depth=depth,
            view_weights=view_weights,
        )
        depth_patchmatch[3] = depths
        depth = depths[-1].detach()
        
        # 上采样：将低分辨率深度 (H/8) 上采样到 (H/2)，上采样因子 4
        depth = F.interpolate(depth, scale_factor=4.0, mode="nearest")
        view_weights = F.interpolate(view_weights, scale_factor=4.0, mode="nearest")

        # Step 3. PatchMatch - 高分辨率 stage1 (H/2, scale=0.5)
        scale = 0.5
        src_features_l = [src_fea[1] for src_fea in src_features]
        intrinsics_l = intrinsics.clone()
        intrinsics_l[:, :, :2] *= scale
        proj = extrinsics.clone()
        proj[:, :, :3, :4] = torch.matmul(intrinsics_l, extrinsics[:, :, :3, :4])
        proj_l = torch.unbind(proj, 1)
        ref_proj, src_proj = proj_l[0], proj_l[1:]
        
        depths, score, view_weights = self.patchmatch_1(
            ref_feature=ref_feature[1],
            src_features=src_features_l,
            ref_proj=ref_proj,
            src_projs=src_proj,
            depth_min=depth_min,
            depth_max=depth_max,
            depth=depth,  # 使用上采样后的初始深度
            view_weights=view_weights,
        )
        depth_patchmatch[1] = depths
        depth = depths[-1].detach()

        del ref_feature, src_features

        # Step 4. Refinement
        depth = self.upsample_net(ref_image, depth, depth_min, depth_max)
        if ref_width != orig_width or ref_height != orig_height:
            depth = F.interpolate(depth, size=[orig_height, orig_width], mode='bilinear', align_corners=False)
        depth_patchmatch[0] = [depth]

        if self.training:
            return depth, torch.empty(0, device=device), depth_patchmatch
        else:
            num_depth = self.patchmatch_num_sample[0] if isinstance(self.patchmatch_num_sample[0], int) else self.patchmatch_num_sample[0]
            score_sum4 = 4 * F.avg_pool3d(
                F.pad(score.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)),
                (4, 1, 1), stride=1, padding=0
            ).squeeze(1)
            depth_index = depth_regression(score, depth_values=torch.arange(num_depth, device=score.device, dtype=torch.float)).long().clamp(0, num_depth - 1)
            photometric_confidence = torch.gather(score_sum4, 1, depth_index)
            photometric_confidence = F.interpolate(photometric_confidence, size=[orig_height, orig_width], mode="nearest").squeeze(1)
            return depth, photometric_confidence, depth_patchmatch

# --- Test 主函数 ---
def main():
    # 设置随机种子，方便复现
    torch.manual_seed(0)

    # 假定 batch_size=1, 图像数 N=3, 图像尺寸 512x512
    batch_size = 1
    num_views = 3
    H, W = 512, 512

    # 构造 dummy 输入图像
    images = [torch.rand(batch_size, 3, H, W) for _ in range(num_views)]
    # 构造 dummy 内参：[B, N, 3, 3] (假设单位矩阵)
    intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1)
    # 构造 dummy 外参：[B, N, 4, 4] (假设单位矩阵扩展)
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1)
    # 构造 dummy 深度范围
    depth_min = torch.tensor([[0.1]])
    depth_max = torch.tensor([[10.0]])

    # 设置 PatchmatchNet 参数（列表长度均为3，对应原网络中三个阶段，但只使用 index=0 和 index=2）
    patchmatch_interval_scale = [0.01, 0.02, 0.03]
    propagation_range = [3, 5, 7]
    patchmatch_iteration = [2, 2, 2]
    patchmatch_num_sample = [64, 48, 32]
    propagate_neighbors = [8, 8, 8]
    evaluate_neighbors = [16, 16, 16]

    # 实例化 PatchmatchNet，并设为评估模式
    net = PatchmatchNet(
        patchmatch_interval_scale,
        propagation_range,
        patchmatch_iteration,
        patchmatch_num_sample,
        propagate_neighbors,
        evaluate_neighbors,
        featureNet='FeatureNet'
    )
    net.eval()

    # 前向传播
    with torch.no_grad():
        depth, photometric_confidence, depth_patchmatch = net(
            images, intrinsics, extrinsics, depth_min, depth_max
        )

    # 输出各部分信息
    print("最终 refined depth shape:", depth.shape)
    print("photometric_confidence shape:", photometric_confidence.shape)
    print("depth_patchmatch keys:", depth_patchmatch.keys())
    for key, d_list in depth_patchmatch.items():
        if isinstance(d_list, list):
            print(f"Stage {key} 的 depth 输出数量: {len(d_list)}")
            if len(d_list) > 0:
                print(f"  第一个输出 shape: {d_list[0].shape}")

if __name__ == '__main__':
    main()
