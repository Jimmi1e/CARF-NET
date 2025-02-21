from torchsummary import summary
# from models.repvit_feature import RepViTNet
from models.liteFeatureNet import LightFeatureNet
from models.net import FeatureNet
# model = RepViTNet(ckpt_path="checkpoints/repvit_m1_5_distill_450e.pth")
model = LightFeatureNet()
summary(model, input_size=(3, 512, 512), batch_size=1, device="cpu")
