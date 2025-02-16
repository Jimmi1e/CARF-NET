import torch
import torch.nn as nn
import torch.nn.functional as F

from models.repvit import repvit_m1_5  

def feature_check(input_size=(3, 512, 512), batch_size=1, device="cpu"):
    model = repvit_m1_5(pretrained=False)
    model = model.to(device)
    
    dummy_input = torch.randn(batch_size, *input_size, device=device)
    print("Input shape:", dummy_input.shape)
    
    features = []
    x = dummy_input
    for idx, layer in enumerate(model.features):
        x = layer(x)
        features.append(x)
        print(f"Layer {idx} output shape: {x.shape}")
        
    return features

if __name__ == '__main__':
    features = feature_check()
