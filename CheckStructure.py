from models import PatchmatchNet, patchmatchnet_loss,TransformerFeature,SwinTransformerV2
from models.net import FeatureNet
from torchsummary import summary
import torch
if __name__ == '__main__':
    
    model=PatchmatchNet(
        patchmatch_interval_scale=[0.005, 0.0125, 0.025],
        propagation_range=[6, 4, 2],
        patchmatch_iteration=[1, 2, 2],
        patchmatch_num_sample=[8, 8, 16],
        propagate_neighbors=[0, 8, 16],
        evaluate_neighbors=[9, 9, 9],
        featureNet='TransformerFeature'
    )
    # 创建测试数据
    images = [torch.rand( 1,3, 512, 512) for _ in range(2)]
    intrinsics = torch.rand(1, 2, 3, 3)
    extrinsics = torch.rand(1, 2, 4, 4)
    depth_min = torch.tensor([0.1])
    depth_max = torch.tensor([10.0])
    output = model(images, intrinsics, extrinsics, depth_min, depth_max)
    print("Depth:", output[0])
    print("Photometric Confidence:", output[1])
    print("Depth Patchmatch:", output[2])
    #-----------------------------------------------#
    # model=TransformerFeature()
    
    # model=SwinTransformerV2()
    # summary(model, input_size=(3, 224, 224), batch_size=1, device="cpu")
    # model = FeatureNet()
    
    # summary(model, input_size=(3, 512, 512), batch_size=1,device="cpu")