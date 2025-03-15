import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple


# ---------------- Dummy Functions and Modules ----------------

def adjust_image_dims(images, intrinsics):
    """Dummy: Assume images are already in correct dimensions."""
    B, C, H, W = images[0].shape
    return images, intrinsics, H, W

class ConvBnReLU(nn.Module):
    """Dummy convolution + BN + ReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Dummy DepthInitialization: returns random depth samples.
class DepthInitialization(nn.Module):
    def __init__(self, patchmatch_num_sample):
        super(DepthInitialization, self).__init__()
        self.patchmatch_num_sample = patchmatch_num_sample
    def forward(self, min_depth, max_depth, height, width, depth_interval_scale, device, depth):
        B = min_depth.size(0)
        return torch.rand(B, self.patchmatch_num_sample, height, width, device=device) * (max_depth - min_depth).view(B, 1, 1, 1) + min_depth.view(B, 1, 1, 1)

# Dummy Propagation: returns the input depth sample.
class Propagation(nn.Module):
    def __init__(self):
        super(Propagation, self).__init__()
    def forward(self, depth_sample, grid):
        return depth_sample

# Dummy Evaluation: returns random depth, score, and view_weights.
class Evaluation(nn.Module):
    def __init__(self, G, Attention_Selection):
        super(Evaluation, self).__init__()
    def forward(self, ref_feature, src_features, ref_proj, src_projs, depth_sample, grid, weight, view_weights, is_inverse):
        B, _, H, W = ref_feature.shape
        new_depth = torch.rand(B, H, W, device=ref_feature.device)
        score = torch.rand(B, 16, H, W, device=ref_feature.device)
        view_weights = torch.rand(B, len(src_features), H, W, device=ref_feature.device)
        return new_depth, score, view_weights

# Dummy FeatureWeightNet: returns ones.
class FeatureWeightNet(nn.Module):
    def __init__(self, evaluate_neighbors, G):
        super(FeatureWeightNet, self).__init__()
    def forward(self, ref_feature, eval_grid):
        B, C, H, W = ref_feature.shape
        return torch.ones(B, 1, H, W, device=ref_feature.device)

def depth_weight(depth_sample, depth_min, depth_max, grid, patchmatch_interval_scale, neighbors):
    B, N, H, W = depth_sample.shape
    return torch.ones(B, N, neighbors, H, W, device=depth_sample.device)

def depth_regression(score, depth_values):
    # Dummy: return argmax over depth dimension.
    return torch.argmax(score, dim=1, keepdim=True).float()

# ---------------- Dummy PatchMatch Module ----------------

class PatchMatch(nn.Module):
    """Dummy PatchMatch module."""
    def __init__(
        self,
        propagation_out_range: int = 2,
        patchmatch_iteration: int = 2,
        patchmatch_num_sample: int = 16,
        patchmatch_interval_scale: float = 0.025,
        num_feature: int = 64,
        G: int = 8,
        propagate_neighbors: int = 16,
        evaluate_neighbors: int = 9,
        stage: int = 3,
        Attention_Selection='None'
    ) -> None:
        super(PatchMatch, self).__init__()
        self.patchmatch_iteration = patchmatch_iteration
        self.patchmatch_interval_scale = patchmatch_interval_scale
        self.propa_num_feature = num_feature
        self.G = G
        self.stage = stage
        self.dilation = propagation_out_range
        self.propagate_neighbors = propagate_neighbors
        self.evaluate_neighbors = evaluate_neighbors

        self.depth_initialization = DepthInitialization(patchmatch_num_sample)
        self.propagation = Propagation()
        self.evaluation = Evaluation(self.G, Attention_Selection)
        self.propa_conv = nn.Conv2d(
            in_channels=self.propa_num_feature,
            out_channels=max(2 * self.propagate_neighbors, 1),
            kernel_size=3,
            stride=1,
            padding=self.dilation,
            dilation=self.dilation,
            bias=True,
        )
        nn.init.constant_(self.propa_conv.weight, 0.0)
        nn.init.constant_(self.propa_conv.bias, 0.0)

        self.eval_conv = nn.Conv2d(
            in_channels=self.propa_num_feature,
            out_channels=2 * self.evaluate_neighbors,
            kernel_size=3,
            stride=1,
            padding=self.dilation,
            dilation=self.dilation,
            bias=True,
        )
        nn.init.constant_(self.eval_conv.weight, 0.0)
        nn.init.constant_(self.eval_conv.bias, 0.0)
        self.feature_weight_net = FeatureWeightNet(self.evaluate_neighbors, self.G)

    def get_grid(self, grid_type: int, batch: int, height: int, width: int, offset: torch.Tensor, device: torch.device) -> torch.Tensor:
        # Dummy grid: zeros.
        num_neighbors = self.evaluate_neighbors if grid_type == 2 else self.propagate_neighbors
        return torch.zeros(batch, num_neighbors, height, width, 2, device=device)

    def forward(
        self,
        ref_feature: torch.Tensor,
        src_features: List[torch.Tensor],
        ref_proj: torch.Tensor,
        src_projs: List[torch.Tensor],
        depth_min: torch.Tensor,
        depth_max: torch.Tensor,
        depth: torch.Tensor,
        view_weights: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        device = ref_feature.device
        batch, _, height, width = ref_feature.size()
        propa_grid = self.get_grid(1, batch, height, width, torch.zeros(batch, 2*self.propagate_neighbors, height*width, device=device), device)
        eval_grid = self.get_grid(2, batch, height, width, torch.zeros(batch, 2*self.evaluate_neighbors, height*width, device=device), device)
        feature_weight = self.feature_weight_net(ref_feature.detach(), eval_grid)
        depth_sample = depth  # may be empty
        score = torch.empty(0, device=device)
        depth_samples = []
        for iter in range(1, self.patchmatch_iteration + 1):
            depth_sample = self.depth_initialization(depth_min, depth_max, height, width, self.patchmatch_interval_scale, device, depth_sample)
            if self.propagate_neighbors > 0 and not (self.stage == 1 and iter == self.patchmatch_iteration):
                depth_sample = self.propagation(depth_sample, propa_grid)
            weight = depth_weight(depth_sample.detach(), depth_min, depth_max, eval_grid.detach(), self.patchmatch_interval_scale, self.evaluate_neighbors)
            weight = weight / torch.sum(weight, dim=2, keepdim=True)
            is_inverse = (self.stage == 1 and iter == self.patchmatch_iteration)
            depth_sample, score, view_weights = self.evaluation(ref_feature, src_features, ref_proj, src_projs, depth_sample, eval_grid, weight, view_weights, is_inverse)
            depth_sample = depth_sample.unsqueeze(1)
            depth_samples.append(depth_sample)
        return depth_samples, score, view_weights

# ---------------- FeatureNet and Refinement ----------------

class FeatureNet(nn.Module):
    """Feature extraction network returning low-res (key 3, H/8) and high-res (key 1, H/2) features."""
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
        # Low-res: H/8
        self.output1 = nn.Conv2d(64, 64, 1, bias=False)
        # High-res: H/2
        self.inner1 = nn.Conv2d(32, 64, 1, bias=True)
        self.inner2 = nn.Conv2d(16, 64, 1, bias=True)
        self.output3 = nn.Conv2d(64, 16, 1, bias=False)
    def forward(self, x: torch.Tensor) -> dict:
        conv1 = self.conv1(self.conv0(x))
        conv4 = self.conv4(self.conv3(self.conv2(conv1)))
        conv7 = self.conv7(self.conv6(self.conv5(conv4)))
        conv10 = self.conv10(self.conv9(self.conv8(conv7)))
        out_low = self.output1(conv10)
        intra_feat = F.interpolate(conv10, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner1(conv7)
        intra_feat = F.interpolate(intra_feat, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner2(conv4)
        out_high = self.output3(intra_feat)
        return {3: out_low, 1: out_high}

class Refinement(nn.Module):
    """Dummy refinement network."""
    def __init__(self):
        super(Refinement, self).__init__()
        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(1, 8, 3, 1, 1)
        self.conv2 = ConvBnReLU(8, 8, 3, 1, 1)
        self.deconv = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.conv3 = ConvBnReLU(16, 8, 3, 1, 1)
        self.res = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, img: torch.Tensor, depth_0: torch.Tensor, depth_min: torch.Tensor, depth_max: torch.Tensor) -> torch.Tensor:
        B = depth_min.size(0)
        depth = (depth_0 - depth_min.view(B,1,1,1))/(depth_max - depth_min).view(B,1,1,1)
        conv0 = self.conv0(img)
        deconv = F.relu(self.bn(self.deconv(self.conv2(self.conv1(depth)))), inplace=True)
        res = self.res(self.conv3(torch.cat((deconv, conv0), dim=1)))
        depth = F.interpolate(depth, scale_factor=2.0, mode="nearest") + res
        return depth*(depth_max-depth_min).view(B,1,1,1)+depth_min.view(B,1,1,1)

# ---------------- PatchmatchNet Module (Modified) ----------------

class PatchmatchNet(nn.Module):
    """PatchmatchNet using only stage3 and stage1 (stage2 removed)."""

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
        """
        Initialize PatchmatchNet.
        Only stage3 and stage1 are used.
        """
        super(PatchmatchNet, self).__init__()

        self.stages = 4  # original stages count
        if featureNet == 'FeatureNet':
            self.feature = FeatureNet()
        elif featureNet == 'TransformerFeature':
            self.feature = TransformerFeature(img_size=image_size)
        elif featureNet == 'RepViTNet':
            self.feature = RepViTNet(ckpt_path="checkpoints/repvit_m1_1_distill_450e.pth")
        elif featureNet == 'LightFeatureNet':
            self.feature = LightFeatureNet()
            apply_pruning(self.feature, amount=0.3)
        self.patchmatch_num_sample = patchmatch_num_sample

        num_features = [16, 32, 64]
        self.propagate_neighbors = propagate_neighbors
        self.evaluate_neighbors = evaluate_neighbors
        self.G = [4, 8, 8]

        # Only instantiate stage1 and stage3.
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

    def forward(
        self,
        images: list,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        depth_min: torch.Tensor,
        depth_max: torch.Tensor,
    ) -> tuple:
        """
        Forward pass.
        1. Extract multi-scale features.
        2. Run stage3 (low-res, key=3) and upsample output.
        3. Run stage1 (high-res, key=1) using upsampled depth.
        4. Refine final depth.
        """
        assert len(images) == intrinsics.size()[1], "Mismatched image and intrinsic counts"
        assert len(images) == extrinsics.size()[1], "Mismatched image and extrinsic counts"
        images, intrinsics, orig_height, orig_width = adjust_image_dims(images, intrinsics)
        ref_image = images[0]

        # Multi-scale feature extraction (FeatureNet returns keys 3 and 1)
        features = [self.feature(img) for img in images]
        del images
        ref_feature, src_features = features[0], features[1:]

        depth_min = depth_min.float()
        depth_max = depth_max.float()
        device = intrinsics.device

        # Let PatchMatch initialize depth internally.
        depth = torch.empty(0, device=device)
        depths = []
        score = torch.empty(0, device=device)
        view_weights = torch.empty(0, device=device)
        depth_patchmatch = {}

        # --- Stage 3: Low-res patchmatch (features key=3, approx. H/8) ---
        scale = 0.125
        src_features_l = [src_fea[3] for src_fea in src_features]
        intrinsics_l = intrinsics.clone()
        intrinsics_l[:, :, :2] *= scale
        proj = extrinsics.clone()
        proj[:, :, :3, :4] = torch.matmul(intrinsics_l, extrinsics[:, :, :3, :4])
        proj_l = torch.unbind(proj, 1)
        ref_proj, src_proj = proj_l[0], proj_l[1:]
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
        # Upsample from H/8 to H/2 (factor=4)
        depth = F.interpolate(depth, scale_factor=4.0, mode="nearest")
        view_weights = F.interpolate(view_weights, scale_factor=4.0, mode="nearest")

        # --- Stage 1: High-res patchmatch (features key=1, approx. H/2) ---
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
            depth=depth,
            view_weights=view_weights,
        )
        depth_patchmatch[1] = depths
        depth = depths[-1].detach()

        del ref_feature, src_features

        # --- Refinement ---
        depth = self.upsample_net(ref_image, depth, depth_min, depth_max)
        if ref_image.size(3) != orig_width or ref_image.size(2) != orig_height:
            depth = F.interpolate(depth, size=[orig_height, orig_width], mode='bilinear', align_corners=False)
        depth_patchmatch[0] = [depth]

        if self.training:
            return depth, torch.empty(0, device=device), depth_patchmatch
        else:
            num_depth = self.patchmatch_num_sample[0]
            score_sum4 = 4 * F.avg_pool3d(
                F.pad(score.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)),
                (4, 1, 1), stride=1, padding=0
            ).squeeze(1)
            depth_index = depth_regression(
                score, depth_values=torch.arange(num_depth, device=score.device, dtype=torch.float)
            ).long().clamp(0, num_depth - 1)
            photometric_confidence = torch.gather(score_sum4, 1, depth_index)
            photometric_confidence = F.interpolate(
                photometric_confidence, size=[orig_height, orig_width], mode="nearest"
            ).squeeze(1)
            return depth, photometric_confidence, depth_patchmatch

# ---------------- Test Script ----------------

def main():
    torch.manual_seed(42)
    batch_size = 1
    num_views = 3
    H, W = 512, 512

    # Create dummy images.
    images = [torch.rand(batch_size, 3, H, W) for _ in range(num_views)]
    # Dummy intrinsics: identity matrices.
    intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1)
    # Dummy extrinsics: identity matrices (4x4).
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1)
    # Dummy depth range.
    depth_min = torch.tensor([[0.1]])
    depth_max = torch.tensor([[10.0]])

    # PatchmatchNet parameters (for 3 stages originally, but stage2 is removed).
    patchmatch_interval_scale = [0.01, 0.02, 0.03]
    propagation_range = [3, 5, 7]
    patchmatch_iteration = [2, 2, 2]
    patchmatch_num_sample = [64, 48, 32]
    propagate_neighbors = [8, 8, 8]
    evaluate_neighbors = [16, 16, 16]

    # Instantiate PatchmatchNet.
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

    with torch.no_grad():
        depth, photometric_confidence, depth_patchmatch = net(
            images, intrinsics, extrinsics, depth_min, depth_max
        )

    print("Refined depth shape:", depth.shape)
    print("Photometric confidence shape:", photometric_confidence.shape)
    print("Depth patchmatch keys:", depth_patchmatch.keys())
    for key in depth_patchmatch:
        for i, d in enumerate(depth_patchmatch[key]):
            print(f"Stage {key}, output {i} shape: {d.shape}")

if __name__ == '__main__':
    main()
