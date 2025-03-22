
from models import PatchmatchNet, patchmatchnet_loss,TransformerFeature,SwinTransformerV2
from models.net import FeatureNet
from models.repvit_feature import RepViTNet
from models.repvit_feature0_9 import RepViTNet09
from torchsummary import summary
import torch
import torchvision.models as models
import torch.nn as nn
if __name__ == '__main__':
    
    model=PatchmatchNet(
        patchmatch_interval_scale=[0.005, 0.0125, 0.025],
        propagation_range=[6, 4, 2],
        patchmatch_iteration=[1, 2, 2],
        patchmatch_num_sample=[8, 8, 16],
        propagate_neighbors=[0, 8, 16],
        evaluate_neighbors=[9, 9, 9],
        use_ARF=True,
        use_CA=True,
        Attention_Selection='Depth',
        featureNet='ResFeatureNet'
        # Use_Cost_reg=True
        # Use_Cost_reg=True
        # Attention_Selection='CBAM'
        # featureNet='RepViTNet09',
        # image_size=(512,640)
        # num_features = [32, 96, 192]
    )
    # for name, param in model.named_parameters():
    #     print(name, param.data.size())
    # 创建测试数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    images = [torch.rand( 1,3, 512, 640) for _ in range(2)]
    intrinsics = torch.rand(1, 2, 3, 3)
    extrinsics = torch.rand(1, 2, 4, 4)
    depth_min = torch.tensor([0.1])
    depth_max = torch.tensor([10.0])
    model.eval()
    output = model(images, intrinsics, extrinsics, depth_min, depth_max)
    print("Depth:", output[0])
    print("Photometric Confidence:", output[1])
    print("Depth Patchmatch:", output[2])
    # allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # 转换为MB

    # print(f"Allocated GPU memory: {allocated_memory} MB")
    #-----------------------------------------------#
    # model =RepViTNet09()
    # model=TransformerFeature()
    
    # model=SwinTransformerV2()
    # summary(model, input_size=(3, 224, 224), batch_size=1, device="cpu")
    # model = FeatureNet()
    # summary(model, input_size=(3, 512, 640), batch_size=1,device="cpu")
    # model = models.resnet18()
    
    # # model = nn.Sequential(*list(model.children())[:-2])
    # resnet=model.layer1
    # summary(resnet, input_size=(3, 512, 640), batch_size=1,device="cpu")
