import torch
import torch.nn as nn
import torch.nn.functional as F

# 请根据你的项目结构调整导入路径
from models.repvit import repvit_m1_5  

def feature_check(input_size=(3, 512, 512), batch_size=1, device="cpu"):
    # 创建 RepViT backbone 模型，注意这里我们只关注 backbone 部分
    # repvit_m1_5 返回的是完整的模型，其中 forward 最后经过 classifier，
    # 我们这里直接使用其 features 部分进行检查。
    model = repvit_m1_5(pretrained=False)
    model = model.to(device)
    
    # 使用一个随机输入
    dummy_input = torch.randn(batch_size, *input_size, device=device)
    print("Input shape:", dummy_input.shape)
    
    features = []
    x = dummy_input
    # 遍历 backbone.features，每经过一层都打印输出尺寸
    for idx, layer in enumerate(model.features):
        x = layer(x)
        features.append(x)
        print(f"Layer {idx} output shape: {x.shape}")
        
    return features

if __name__ == '__main__':
    features = feature_check()
