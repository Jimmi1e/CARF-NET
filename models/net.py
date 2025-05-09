from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import ConvBnReLU, depth_regression,CoordAtt,ResidualBlock
from .patchmatch import PatchMatch
from .swin_transformer_v2 import PatchEmbed,BasicLayer,PatchMerging
from .repvit_feature import RepViTNet
from .repvit_feature11 import RepViTNet11
from .repvit_feature0_9 import RepViTNet09
from .FMT import FMT_with_pathway
from .dcn import DCN
class TransformerFeature(nn.Module):
    """Transformer Feature Network: to extract features of transformed images from each view"""
    def __init__(self,img_size=512,window_size=8, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,use_checkpoint=False,
                 pretrained_window_sizes=[0, 0, 0, 0]):
        """Initialize different layers in the network"""

        super(TransformerFeature, self).__init__()
        self.embed_dim=96
        self.depths=[2,2,6,2]
        self.num_heads=[3, 6, 12, 24]
        self.mlp_ratio = mlp_ratio
        self.num_layers=len(self.depths)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=2, in_chans=3, embed_dim=self.embed_dim,
            norm_layer=nn.LayerNorm )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(self.embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=self.depths[i_layer],
                               num_heads=self.num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                               norm_layer=nn.LayerNorm,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)
        self.output1 = nn.Conv2d(self.embed_dim*8, 64, 1, bias=False)
        self.inner1 = nn.Conv2d(self.embed_dim*4, 64, 1, bias=True)
        self.inner2 = nn.Conv2d(self.embed_dim*2, 64, 1, bias=True)
        self.output2 = nn.Conv2d(64, 32, 1, bias=False)
        self.output3 = nn.Conv2d(64, 16, 1, bias=False)
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Forward method

        Args:
            x: images from a single view, in the shape of [B, C, H, W]. Generally, C=3

        Returns:
            output_feature: a python dictionary contains extracted features from stage 1 to stage 3
                keys are 1, 2, and 3
        """
        output_feature: Dict[int, torch.Tensor] = {}
        x = self.patch_embed(x)
        feature_list=[]
        for i,layer in enumerate(self.layers):
            x = layer(x)
            B, L, C = x.shape
            if i == len(self.layers) - 1:
                H, W = self.patch_embed.patches_resolution[0] // (2 ** (i)), self.patch_embed.patches_resolution[1] // (2 ** (i)) 
            else:
                H, W = self.patch_embed.patches_resolution[0] // (2 ** (i+1)), self.patch_embed.patches_resolution[1] // (2 ** (i+1)) 
            x_reshaped = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            x_reshaped=F.interpolate(x_reshaped, scale_factor=2.0, mode="bilinear", align_corners=False) 
            feature_list.append(x_reshaped)
        output_feature[3] = self.output1(feature_list[-1])
        # print(output_feature[3].size())
        intra_feat = F.interpolate(output_feature[3], scale_factor=2.0, mode="bilinear", align_corners=False) 
        inner1=self.inner1(feature_list[-3])
        output_feature[2] = self.output2(intra_feat+inner1)
        # print(output_feature[2].size())
        intra_feat = F.interpolate(
            intra_feat, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner2(feature_list[-4])
        output_feature[1] = self.output3(intra_feat)
        # print(output_feature[1].size())
        return output_feature
class ResFeatureNet(nn.Module):
    def __init__(self,use_ARF=False,use_CA=False):
        """Initialize different layers in the network"""

        super(ResFeatureNet, self).__init__()
        self.use_CA=use_CA
        self.in_planes = 8
        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.Res_layer1 = self._make_layer(16, stride=2)#in 8 out 16
        self.Res_layer2 = self._make_layer(32, stride=2)# in 16 out 32
        self.Res_layer3 = self._make_layer(64, stride=2)# in 32 out 64
        if use_ARF:
            self.output1 = nn.Sequential(
                    ConvBnReLU(64, 64, 1,1,0),
                    # DCN(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    # nn.BatchNorm2d(64),
                    # nn.ReLU(inplace=True),
                    # DCN(in_channels=64, out_channels=64, kernel_size=3,stride=1, padding=1),
                    # nn.BatchNorm2d(64),
                    # nn.ReLU(inplace=True),
                    DCN(in_channels=64, out_channels=64, kernel_size=3,stride=1, padding=1))
            self.output2 = nn.Sequential(
                    ConvBnReLU(64, 64, 3,1,1),
                    # DCN(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    # nn.BatchNorm2d(64),
                    # nn.ReLU(inplace=True),
                    # DCN(in_channels=64, out_channels=64, kernel_size=3,stride=1, padding=1),
                    # nn.BatchNorm2d(64),
                    # nn.ReLU(inplace=True),
                    DCN(in_channels=64, out_channels=32, kernel_size=3,stride=1, padding=1))
            self.output3 = nn.Sequential(
                    ConvBnReLU(64, 64, 3,1,1),
                    # DCN(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    # nn.BatchNorm2d(64),
                    # nn.ReLU(inplace=True),
                    # DCN(in_channels=64, out_channels=64, kernel_size=3,stride=1, padding=1),
                    # nn.BatchNorm2d(64),
                    # nn.ReLU(inplace=True),
                    DCN(in_channels=64, out_channels=16, kernel_size=3,stride=1, padding=1))
        else:
            self.output1 = nn.Conv2d(64, 64, 1, bias=False)
            self.output2 = nn.Conv2d(64, 32, 1, bias=False)
            self.output3 = nn.Conv2d(64, 16, 1, bias=False)
        if use_CA:
            self.ca1 = CoordAtt(64, 64)
            self.ca2 = CoordAtt(32, 32)
            self.ca3 = CoordAtt(16, 16)
        self.inner1 = nn.Conv2d(32, 64, 1, bias=True)
        self.inner2 = nn.Conv2d(16, 64, 1, bias=True)
    def _make_layer(self, dim, stride=1):   
        layer1 = ResidualBlock(self.in_planes, dim, stride=stride)
        layer2 = ResidualBlock(dim, dim)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Forward method

        Args:
            x: images from a single view, in the shape of [B, C, H, W]. Generally, C=3

        Returns:
            output_feature: a python dictionary contains extracted features from stage 1 to stage 3
                keys are 1, 2, and 3
        """
        output_feature: Dict[int, torch.Tensor] = {}
        
        feature1 = self.Res_layer1(self.conv0(x))
        feature2 = self.Res_layer2(feature1)

        feature3 = self.Res_layer3(feature2)
        # conv10 = self.conv10(self.conv9(self.conv8(conv7)))
        
        # output_feature[3] = self.out1(conv10)
        output_feature[3] = self.output1(feature3)
        
        if self.use_CA:
            output_feature[3] = self.ca1(output_feature[3])
        # print(output_feature[3].size())
        intra_feat = F.interpolate(feature3, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner1(feature2)
        del feature2
        del feature3
        # output_feature[2] = self.out2(intra_feat) 
        
        output_feature[2] = self.output2(intra_feat) 
        if self.use_CA:
            output_feature[2] = self.ca2(output_feature[2])
        # print(output_feature[2].size()) 
        intra_feat = F.interpolate(
            intra_feat, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner2(feature1)
        del feature1
        
        # output_feature[1] = self.out3(intra_feat) 
        output_feature[1] = self.output3(intra_feat)
        if self.use_CA:
            output_feature[1] = self.ca3(output_feature[1])
        # print(output_feature[1].size())
        del intra_feat

        return output_feature
class FeatureNet(nn.Module):
    """Feature Extraction Network: to extract features of original images from each view"""

    def __init__(self,use_ARF=False,use_CA=False):
        """Initialize different layers in the network"""

        super(FeatureNet, self).__init__()
        self.use_CA=use_CA
        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        # [B,8,H,W]
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)
        # [B,16,H/2,W/2]
        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)
        # [B,32,H/4,W/4]
        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.conv7 = ConvBnReLU(32, 32, 3, 1, 1)
        # [B,64,H/8,W/8]
        self.conv8 = ConvBnReLU(32, 64, 5, 2, 2)
        self.conv9 = ConvBnReLU(64, 64, 3, 1, 1)
        self.conv10 = ConvBnReLU(64, 64, 3, 1, 1)

        if use_ARF:
            self.output1 = nn.Sequential(
                    ConvBnReLU(64, 64, 1,1,0),
                    # DCN(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    # nn.BatchNorm2d(64),
                    # nn.ReLU(inplace=True),
                    DCN(in_channels=64, out_channels=64, kernel_size=3,stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    DCN(in_channels=64, out_channels=64, kernel_size=3,stride=1, padding=1))
            self.output2 = nn.Sequential(
                    ConvBnReLU(64, 64, 3,1,1),
                    # DCN(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    # nn.BatchNorm2d(64),
                    # nn.ReLU(inplace=True),
                    DCN(in_channels=64, out_channels=64, kernel_size=3,stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    DCN(in_channels=64, out_channels=32, kernel_size=3,stride=1, padding=1))
            self.output3 = nn.Sequential(
                    ConvBnReLU(64, 64, 3,1,1),
                    # DCN(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    # nn.BatchNorm2d(64),
                    # nn.ReLU(inplace=True),
                    DCN(in_channels=64, out_channels=64, kernel_size=3,stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    DCN(in_channels=64, out_channels=16, kernel_size=3,stride=1, padding=1))
        else:
            self.output1 = nn.Conv2d(64, 64, 1, bias=False)
            self.output2 = nn.Conv2d(64, 32, 1, bias=False)
            self.output3 = nn.Conv2d(64, 16, 1, bias=False)
        if use_CA:
            self.ca1 = CoordAtt(64, 64)
            self.ca2 = CoordAtt(32, 32)
            self.ca3 = CoordAtt(16, 16)
        self.inner1 = nn.Conv2d(32, 64, 1, bias=True)
        self.inner2 = nn.Conv2d(16, 64, 1, bias=True)

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Forward method

        Args:
            x: images from a single view, in the shape of [B, C, H, W]. Generally, C=3

        Returns:
            output_feature: a python dictionary contains extracted features from stage 1 to stage 3
                keys are 1, 2, and 3
        """
        output_feature: Dict[int, torch.Tensor] = {}
        
        conv1 = self.conv1(self.conv0(x))
        conv4 = self.conv4(self.conv3(self.conv2(conv1)))

        conv7 = self.conv7(self.conv6(self.conv5(conv4)))
        conv10 = self.conv10(self.conv9(self.conv8(conv7)))
        
        # output_feature[3] = self.out1(conv10)
        
        output_feature[3] = self.output1(conv10)
        if self.use_CA:
            output_feature[3] = self.ca1(output_feature[3])
        # print(output_feature[3].size())
        intra_feat = F.interpolate(conv10, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner1(conv7)
        del conv7
        del conv10
        # output_feature[2] = self.out2(intra_feat) 
        
        output_feature[2] = self.output2(intra_feat) 
        if self.use_CA:
            output_feature[2] = self.ca2(output_feature[2])
        # print(output_feature[2].size()) 
        intra_feat = F.interpolate(
            intra_feat, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner2(conv4)
        del conv4
        # output_feature[1] = self.out3(intra_feat) 
        
        output_feature[1] = self.output3(intra_feat)
        if self.use_CA:
            output_feature[1] = self.ca3(output_feature[1])
        # print(output_feature[1].size())
        del intra_feat

        return output_feature


class Refinement(nn.Module):
    """Depth map refinement network"""

    def __init__(self):
        """Initialize"""

        super(Refinement, self).__init__()

        # img: [B,3,H,W]
        self.conv0 = ConvBnReLU(in_channels=3, out_channels=8)
        # depth map:[B,1,H/2,W/2]
        self.conv1 = ConvBnReLU(in_channels=1, out_channels=8)
        self.conv2 = ConvBnReLU(in_channels=8, out_channels=8)
        self.deconv = nn.ConvTranspose2d(
            in_channels=8, out_channels=8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
        )

        self.bn = nn.BatchNorm2d(8)
        self.conv3 = ConvBnReLU(in_channels=16, out_channels=8)
        self.res = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1, bias=False)

    def forward(
        self, img: torch.Tensor, depth_0: torch.Tensor, depth_min: torch.Tensor, depth_max: torch.Tensor
    ) -> torch.Tensor:
        """Forward method

        Args:
            img: input reference images (B, 3, H, W)
            depth_0: current depth map (B, 1, H//2, W//2)
            depth_min: pre-defined minimum depth (B, )
            depth_max: pre-defined maximum depth (B, )

        Returns:
            depth: refined depth map (B, 1, H, W)
        """

        batch_size = depth_min.size()[0]
        # pre-scale the depth map into [0,1]
        depth = (depth_0 - depth_min.view(batch_size, 1, 1, 1)) / (depth_max - depth_min).view(batch_size, 1, 1, 1)

        conv0 = self.conv0(img)
        deconv = F.relu(self.bn(self.deconv(self.conv2(self.conv1(depth)))), inplace=True)
        # depth residual
        res = self.res(self.conv3(torch.cat((deconv, conv0), dim=1)))
        del conv0
        del deconv

        depth = F.interpolate(depth, scale_factor=2.0, mode="nearest") + res
        # convert the normalized depth back
        return depth * (depth_max - depth_min).view(batch_size, 1, 1, 1) + depth_min.view(batch_size, 1, 1, 1)


class PatchmatchNet(nn.Module):
    """ Implementation of complete structure of PatchmatchNet"""

    def __init__(
        self,
        patchmatch_interval_scale: List[float],
        propagation_range: List[int],
        patchmatch_iteration: List[int],
        patchmatch_num_sample: List[int],
        propagate_neighbors: List[int],
        evaluate_neighbors: List[int],
        use_ARF=False,
        use_FMT=False,
        use_CA=False,
        featureNet='FeatureNet',
        image_size=(512,512),
        num_features = [16, 32, 64],
        Attention_Selection='None',
        Attention_Selection_FWN='None',
        Use_Cost_reg=False
    ) -> None:
        """Initialize modules in PatchmatchNet

        Args:
            patchmatch_interval_scale: depth interval scale in patchmatch module
            propagation_range: propagation range
            patchmatch_iteration: patchmatch iteration number
            patchmatch_num_sample: patchmatch number of samples
            propagate_neighbors: number of propagation neighbors
            evaluate_neighbors: number of propagation neighbors for evaluation
        """
        super(PatchmatchNet, self).__init__()
        self.use_FMT = use_FMT
        self.stages = 4
        if featureNet == 'FeatureNet':
            self.feature = FeatureNet(use_ARF=use_ARF,use_CA=use_CA)
        elif featureNet == 'TransformerFeature':
            self.feature = TransformerFeature(img_size=image_size)
        elif featureNet=='RepViTNet':
            self.feature = RepViTNet(ckpt_path="checkpoints/repvit_m1_5_distill_450e.pth")# new add Jiaxi
        elif featureNet=="RepViTNet11":
            self.feature = RepViTNet11(ckpt_path="checkpoints/repvit_m1_1_distill_450e.pth")
        elif featureNet=="RepViTNet09":
            self.feature = RepViTNet09(ckpt_path="checkpoints/repvit_m0_9_distill_450e.pth")
        elif featureNet=="ResFeatureNet":
            self.feature = ResFeatureNet(use_ARF=use_ARF,use_CA=use_CA)
        self.patchmatch_num_sample = patchmatch_num_sample

        # num_features = [16, 32, 64]  #move to init
        if use_FMT:
            self.FMT_with_pathway = FMT_with_pathway()
        self.propagate_neighbors = propagate_neighbors
        self.evaluate_neighbors = evaluate_neighbors
        # number of groups for group-wise correlation
        self.G = [4, 8, 8]

        for i in range(self.stages - 1):
            patchmatch = PatchMatch(
                propagation_out_range=propagation_range[i],
                patchmatch_iteration=patchmatch_iteration[i],
                patchmatch_num_sample=patchmatch_num_sample[i],
                patchmatch_interval_scale=patchmatch_interval_scale[i],
                num_feature=num_features[i],
                G=self.G[i],
                propagate_neighbors=self.propagate_neighbors[i],
                evaluate_neighbors=evaluate_neighbors[i],
                stage=i + 1,
                Attention_Selection=Attention_Selection,
                Attention_Selection_FWN=Attention_Selection_FWN,
                Use_Cost_reg=Use_Cost_reg
            )
            setattr(self, f"patchmatch_{i+1}", patchmatch)

        self.upsample_net = Refinement()

    def forward(
        self,
        images: List[torch.Tensor],
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        depth_min: torch.Tensor,
        depth_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, List[torch.Tensor]]]:
        """Forward method for PatchMatchNet

        Args:
            images: N images (B, 3, H, W) stored in list
            intrinsics: intrinsic 3x3 matrices for all images (B, N, 3, 3)
            extrinsics: extrinsic 4x4 matrices for all images (B, N, 4, 4)
            depth_min: minimum virtual depth (B, 1)
            depth_max: maximum virtual depth (B, 1)

        Returns:
            output tuple of PatchMatchNet, containing refined depthmap, depth patchmatch, and photometric confidence.
        """
        assert len(images) == intrinsics.size()[1], "Different number of images and intrinsic matrices"
        assert len(images) == extrinsics.size()[1], 'Different number of images and extrinsic matrices'
        images, intrinsics, orig_height, orig_width = adjust_image_dims(images, intrinsics)
        ref_image = images[0]
        _, _, ref_height, ref_width = ref_image.size()

        # step 1. Multi-scale feature extraction
        features: List[Dict[int, torch.Tensor]] = []
        for img in images:
            output_feature = self.feature(img)
            features.append(output_feature)
        del images
        ## Add FMT
        if self.use_FMT:
            features = self.FMT_with_pathway(features)
        ref_feature, src_features = features[0], features[1:]

        depth_min = depth_min.float()
        depth_max = depth_max.float()

        # step 2. Learning-based patchmatch
        device = intrinsics.device
        depth = torch.empty(0, device=device)
        depths: List[torch.Tensor] = []
        score = torch.empty(0, device=device)
        view_weights = torch.empty(0, device=device)
        depth_patchmatch: Dict[int, List[torch.Tensor]] = {}

        scale = 0.125
        for stage in range(self.stages - 1, 0, -1):
            src_features_l = [src_fea[stage] for src_fea in src_features]

            # Create projection matrix for specific stage
            intrinsics_l = intrinsics.clone()
            intrinsics_l[:, :, :2] *= scale
            proj = extrinsics.clone()
            proj[:, :, :3, :4] = torch.matmul(intrinsics_l, extrinsics[:, :, :3, :4])
            proj_l = torch.unbind(proj, 1)
            ref_proj, src_proj = proj_l[0], proj_l[1:]
            scale *= 2.0

            # Need conditional since TorchScript only allows "getattr" access with string literals
            if stage == 3:
                depths, score, view_weights = self.patchmatch_3(
                    ref_feature=ref_feature[stage],
                    src_features=src_features_l,
                    ref_proj=ref_proj,
                    src_projs=src_proj,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    depth=depth,
                    view_weights=view_weights,
                )
            elif stage == 2:
                depths, score, view_weights = self.patchmatch_2(
                    ref_feature=ref_feature[stage],
                    src_features=src_features_l,
                    ref_proj=ref_proj,
                    src_projs=src_proj,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    depth=depth,
                    view_weights=view_weights,
                )
            elif stage == 1:
                depths, score, view_weights = self.patchmatch_1(
                    ref_feature=ref_feature[stage],
                    src_features=src_features_l,
                    ref_proj=ref_proj,
                    src_projs=src_proj,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    depth=depth,
                    view_weights=view_weights,
                )

            depth_patchmatch[stage] = depths
            depth = depths[-1].detach()

            if stage > 1:
                # upsampling the depth map and pixel-wise view weight for next stage
                depth = F.interpolate(depth, scale_factor=2.0, mode="nearest")
                view_weights = F.interpolate(view_weights, scale_factor=2.0, mode="nearest")

        del ref_feature
        del src_features

        # step 3. Refinement
        depth = self.upsample_net(ref_image, depth, depth_min, depth_max)
        if ref_width != orig_width or ref_height != orig_height:
            depth = F.interpolate(depth, size=[orig_height, orig_width], mode='bilinear', align_corners=False)
        depth_patchmatch[0] = [depth]

        if self.training:
            return depth, torch.empty(0, device=device), depth_patchmatch
        else:
            num_depth = self.patchmatch_num_sample[0]
            score_sum4 = 4 * F.avg_pool3d(
                F.pad(score.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0
            ).squeeze(1)
            # [B, 1, H, W]
            depth_index = depth_regression(
                score, depth_values=torch.arange(num_depth, device=score.device, dtype=torch.float)
            ).long().clamp(0, num_depth - 1)
            photometric_confidence = torch.gather(score_sum4, 1, depth_index)
            photometric_confidence = F.interpolate(
                photometric_confidence, size=[orig_height, orig_width], mode="nearest").squeeze(1)

            return depth, photometric_confidence, depth_patchmatch


def adjust_image_dims(
        images: List[torch.Tensor], intrinsics: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, int, int]:
    # stretch or compress image slightly to ensure width and height are multiples of 8
    _, _, ref_height, ref_width = images[0].size()
    for i in range(len(images)):
        _, _, height, width = images[i].size()
        new_height = int(round(height / 8)) * 8
        new_width = int(round(width / 8)) * 8
        if new_width != width or new_height != height:
            intrinsics[:, i, 0] *= new_width / width
            intrinsics[:, i, 1] *= new_height / height
            images[i] = nn.functional.interpolate(
                images[i], size=[new_height, new_width], mode='bilinear', align_corners=False)

    return images, intrinsics, ref_height, ref_width


def patchmatchnet_loss(
    depth_patchmatch: Dict[int, List[torch.Tensor]],
    depth_gt: List[torch.Tensor],
    mask: List[torch.Tensor],
) -> torch.Tensor:
    """Patchmatch Net loss function

    Args:
        depth_patchmatch: depth map predicted by patchmatch net
        depth_gt: ground truth depth map
        mask: mask for filter valid points

    Returns:
        loss: result loss value
    """
    loss = 0
    for i in range(0, 4):
        gt_depth = depth_gt[i][mask[i].bool()]
        for depth in depth_patchmatch[i]:
            loss = loss + F.smooth_l1_loss(depth[mask[i].bool()], gt_depth, reduction="mean")

    return loss
def computer_normal(depth):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=depth.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=depth.device).unsqueeze(0).unsqueeze(0)
    
    dzdx = F.conv2d(depth, sobel_x, padding=1)  # X 方向梯度
    dzdy = F.conv2d(depth, sobel_y, padding=1)  # Y 方向梯度
    normals = torch.cat([-dzdx, -dzdy, torch.ones_like(depth)], dim=1)
    normals = F.normalize(normals,p=2, dim=1)  # 归一化
    return normals
def depth_normal_loss(
    depth_patchmatch: Dict[int, List[torch.Tensor]],
    depth_gt: List[torch.Tensor],
    mask: List[torch.Tensor],
) -> torch.Tensor:
    """Patchmatch Net loss function

    Args:
        depth_patchmatch: depth map predicted by patchmatch net
        depth_gt: ground truth depth map
        mask: mask for filter valid points

    Returns:
        loss: result loss value
    """
    loss = 0
    for i in range(0, 4):
        #print(depth_gt[i].shape)
        gt_depth = depth_gt[i][mask[i].bool()]
        gt_normal = computer_normal(depth_gt[i])
        for depth in depth_patchmatch[i]:
            predit_depth=depth[mask[i].bool()]
            predit_normal = computer_normal(depth)
            normal_loss = 1 - torch.mean(torch.sum(predit_normal * gt_normal, dim=1))
            depth_loss=F.smooth_l1_loss(predit_depth, gt_depth, reduction="mean")
            #print('normal',normal_loss,'Depth',depth_loss)
            loss = loss +depth_loss +normal_loss*10

    return loss
def entropy_loss(prob_volume, depth_gt, mask, depth_value, return_prob_map=False):
    # from AA
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6

    shape = depth_gt.shape          # B,H,W

    depth_num = depth_value.shape[1]
    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)     # B,N,H,W
    else:
        depth_value_mat = depth_value

    gt_index_image = torch.argmin(torch.abs(depth_value_mat-depth_gt.unsqueeze(1)), dim=1)

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1) # B, 1, H, W

    # gt index map -> gt one hot volume (B x 1 x H x W )
    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)

    # cross entropy image (B x D X H x W)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-6), dim=1).squeeze(1) # B, 1, H, W

    # masked cross entropy loss
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image) # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num) # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(prob_volume, dim=1)[0] # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map, photometric_confidence
    return masked_cross_entropy, wta_depth_map
def focal_loss_bld(inputs, depth_gt_ms, mask_ms, depth_interval, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_entropy = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        prob_volume = stage_inputs["prob_volume"]
        depth_values = stage_inputs["depth_values"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        entropy_weight = 2.0
        entro_loss, depth_entropy = entropy_loss(prob_volume, depth_gt, mask, depth_values)
        entro_loss = entro_loss * entropy_weight
        depth_loss = F.smooth_l1_loss(depth_entropy[mask], depth_gt[mask], reduction='mean')
        total_entropy += entro_loss

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * entro_loss
        else:
            total_loss += entro_loss

    abs_err = (depth_gt_ms['stage3'] - inputs["stage3"]["depth"]).abs()
    abs_err_scaled = abs_err /(depth_interval *192./128.)
    mask = mask_ms["stage3"]
    mask = mask > 0.5
    epe = abs_err_scaled[mask].mean()
    less1 = (abs_err_scaled[mask] < 1.).to(depth_gt_ms['stage3'].dtype).mean()
    less3 = (abs_err_scaled[mask] < 3.).to(depth_gt_ms['stage3'].dtype).mean()

    return total_loss, depth_loss, epe, less1, less3