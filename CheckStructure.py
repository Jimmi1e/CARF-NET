from models import PatchmatchNet, patchmatchnet_loss,TransformerFeature,SwinTransformerV2
from models.net import FeatureNet
from torchsummary import summary

if __name__ == '__main__':
    
    model=TransformerFeature()
    # model=SwinTransformerV2()
    # summary(model, input_size=(3, 224, 224), batch_size=1, device="cpu")
    # model = FeatureNet()
    summary(model, input_size=(3, 512, 512), batch_size=1,device="cpu")