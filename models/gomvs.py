import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import ConvBnReLU3D, homo_warping  # 请确保 module.py 中有: homo_warping = differentiable_warping
from .compute_normal import depth2normal
from .gca_module import GCACostRegNet

Align_Corners_Range = False

def depth_wta(p, depth_values):
    wta_index_map = torch.argmax(p, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_values, 1, wta_index_map).squeeze(1)
    return wta_depth_map

class PixelwiseNet(nn.Module):
    def __init__(self):
        super(PixelwiseNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels=1, out_channels=16, kernel_size=1, stride=1, pad=0)
        self.conv1 = ConvBnReLU3D(in_channels=16, out_channels=8, kernel_size=1, stride=1, pad=0)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()
    def forward(self, x1):
        x1 = self.conv2(self.conv1(self.conv0(x1))).squeeze(1)  # [B, D, H, W]
        output = self.output(x1)
        output = torch.max(output, dim=1, keepdim=True)[0]
        return output

class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.pixel_wise_net = PixelwiseNet()
    def forward(self,
                features,        # list: [ref_feature, src_feature1, src_feature2, ...]
                proj_matrices,   # Tensor: (B, N, 2, 4, 4)
                depth_values,    # Tensor: [B, D, H, W]
                num_depth,       # int: D
                cost_regularization,  # instance of GCACostRegNet
                prob_volume_init=None,
                normal=None,
                stage_intric=None,
                view_weights=None):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, f"depth_values.shape[1]: {depth_values.shape[1]}  num_depth: {num_depth}"
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        if view_weights is None:
            view_weight_list = []
        similarity_sum = 0
        pixel_wise_weight_sum = 1e-5
        for i, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
            similarity = (warped_volume * ref_feature.unsqueeze(2)).mean(1, keepdim=True)
            if view_weights is None:
                view_weight = self.pixel_wise_net(similarity)
                view_weight_list.append(view_weight)
            else:
                view_weight = view_weights[:, i:i+1]
            similarity_sum += similarity * view_weight.unsqueeze(1)
            pixel_wise_weight_sum += view_weight.unsqueeze(1)
            del warped_volume
        similarity = similarity_sum / pixel_wise_weight_sum  # [B, 1, D, H, W]
        similarity_prob = F.softmax(similarity.squeeze(1), dim=1)
        similarity_depth = depth_wta(similarity_prob, depth_values=depth_values)
        cost_reg = cost_regularization(similarity, depth_values, normal, stage_intric)
        prob_volume_pre = cost_reg.squeeze(1)
        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init
        prob_volume = torch.exp(F.log_softmax(prob_volume_pre, dim=1))
        depth = depth_wta(prob_volume, depth_values=depth_values)
        with torch.no_grad():
            photometric_confidence = torch.max(prob_volume, dim=1)[0]
        if view_weights is None:
            view_weights = torch.cat(view_weight_list, dim=1)
            return {"depth": depth, "similarity_depth": similarity_depth,
                    "prob_volume": prob_volume, "depth_values": depth_values,
                    "photometric_confidence": photometric_confidence}, view_weights.detach()
        else:
            return {"depth": depth, "similarity_depth": similarity_depth,
                    "prob_volume": prob_volume, "depth_values": depth_values,
                    "photometric_confidence": photometric_confidence}

class GoMVS(nn.Module):
    """
    修改后的 GoMVS 模块：
    接受外部传入的参考视角和源视角全分辨率特征，以及由 PatchMatch 得到的初步深度 (depth_init)。
    将 depth_init 上采样到全分辨率后，在其附近生成候选深度，
    构造 dummy 投影矩阵（基于参考视角 extrinsics），并调用 DepthNet 进行成本体积正则化与深度回归，
    同时传入 intri（参考视角内参），以便 GCACostRegNet 正确计算。
    """
    def __init__(self, ndepth=16, depth_offset=0.01, grad_method="detach", cr_base_chs=8, mode="train"):
        super(GoMVS, self).__init__()
        self.ndepth = ndepth
        self.depth_offset = depth_offset
        self.grad_method = grad_method
        self.mode = mode
        self.cost_regularization = GCACostRegNet(in_channels=1, base_channels=8)
        self.DepthNet = DepthNet()
    def forward(self, ref_feature, src_features, ref_proj, src_proj, depth_init, intri, normal_init=None):
        # depth_init: [B, 1, H_low, W_low]，初步深度（低分辨率）
        B, _, H_low, W_low = depth_init.shape
        # 将初步深度上采样到全分辨率，与 ref_feature 一致
        _, _, H_full, W_full = ref_feature.shape
        depth_init_full = F.interpolate(depth_init, size=(H_full, W_full), mode='bilinear', align_corners=False)
        
        # 如果未提供法向信息，则生成默认法向 [0,0,1]
        if normal_init is None:
            normal_init = torch.zeros(B, 3, H_full, W_full, device=ref_feature.device)
            normal_init[:, 2, :, :] = 1.0
        
        D = self.ndepth
        # 生成候选深度：在上采样后的 depth_init_full 附近均匀采样 D 个候选深度
        offsets = torch.linspace(-self.depth_offset, self.depth_offset, steps=D, device=depth_init.device).view(1, D, 1, 1)
        depth_candidates = depth_init_full.repeat(1, D, 1, 1) + offsets  # [B, D, H_full, W_full]
        
        # 构造 dummy 投影矩阵
        # 假设 N = 1 + len(src_features)
        N = 1 + len(src_features)
        # ref_proj 应该为 [B, 1, 4, 4]
        dummy_proj = ref_proj.repeat(1, N, 1, 1)  # [B, N, 4, 4]
        dummy_proj = dummy_proj.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # [B, N, 2, 4, 4]
        
        features = [ref_feature] + src_features
        # 将 intri（参考视角内参）传入作为 stage_intric
        outputs, _ = self.DepthNet(features, dummy_proj, depth_candidates, num_depth=D,
                                   cost_regularization=self.cost_regularization, normal=normal_init,
                                   stage_intric=intri, view_weights=None)
        wta_index_map = torch.argmax(outputs['prob_volume'], dim=1, keepdim=True).type(torch.long)
        depth_refined = torch.gather(outputs['depth_values'], 1, wta_index_map).squeeze(1)
        return {"depth": depth_refined, "prob_volume": outputs['prob_volume'], "depth_values": depth_candidates}, torch.empty(0)

# get_depth_range_samples 如前定义，此处略...

def get_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, device, dtype, shape,
                           max_depth=192.0, min_depth=0.0, use_inverse_depth=False):
    if cur_depth.dim() == 2:
        cur_depth_min = cur_depth[:, 0]
        cur_depth_max = cur_depth[:, -1]
        if not use_inverse_depth:
            new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)
            depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype).reshape(1, -1) * new_interval.unsqueeze(1))
            depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2])
        else:
            depth_range_samples = cur_depth.repeat(1, 1, shape[1], shape[2])
    else:
        depth_range_samples = cur_depth
    return depth_range_samples
