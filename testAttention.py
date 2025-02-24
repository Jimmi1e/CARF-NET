import torch
from models.module import CBAM3D, DepthAxialAttention3D

if __name__ == '__main__':
    # 构造一个随机输入：例如 B=2, C=32, D=8, H=64, W=64
    dummy_input = torch.randn(2, 32, 8, 64, 64)
    cbam = CBAM3D(in_channels=32, reduction_ratio=8, spatial_kernel=7)
    
    # 前向传播，得到输出
    output = cbam(dummy_input)
    
    # 验证输出形状与输入一致
    print("Input shape: ", dummy_input.shape)
    print("Output shape:", output.shape)
    
    # 验证注意力机制的数值范围
    with torch.no_grad():
        # 获取通道注意力
        avg_out = cbam.mlp(cbam.avg_pool(dummy_input))
        max_out = cbam.mlp(cbam.max_pool(dummy_input))
        channel_att = cbam.sigmoid_channel(avg_out + max_out)
        print("Channel attention min/max:", channel_att.min().item(), channel_att.max().item())
        
        # 计算空间注意力
        x_channel = dummy_input * channel_att
        avg_out_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_out_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_cat = torch.cat([avg_out_spatial, max_out_spatial], dim=1)
        spatial_att = cbam.sigmoid_spatial(cbam.conv_spatial(spatial_cat))
        print("Spatial attention min/max:", spatial_att.min().item(), spatial_att.max().item())
    
    # 检查梯度是否能够正常反向传播
    dummy_input.requires_grad = True
    output = cbam(dummy_input)
    loss = output.sum()
    loss.backward()
    print("Gradient check passed: ", dummy_input.grad is not None)

# if __name__ == '__main__':
#     # 构造一个随机输入，假设输入尺寸为 [B, C, D, H, W]
#     dummy = torch.randn(2, 32, 8, 64, 64)
#     model = DepthAxialAttention3D(in_channels=32, reduction_ratio=8)
    
#     # 前向传播
#     output = model(dummy)
#     print("Input shape: ", dummy.shape)
#     print("Output shape:", output.shape)
    
#     # 检查通道注意力部分的数值范围（打印最小、最大值）
#     with torch.no_grad():
#         ca = model.channel_attention(dummy)
#         print("Channel attention range: ", ca.min().item(), ca.max().item())
        
#         # 经过通道注意力后
#         x_ca = dummy * ca
#         z_feat = model.depth_conv(x_ca)
#         sa = model.spatial_attention(z_feat)
#         print("Spatial attention range: ", sa.min().item(), sa.max().item())
    
#     # 梯度检查：确保可以反向传播
#     dummy.requires_grad = True
#     out = model(dummy)
#     loss = out.sum()
#     loss.backward()
#     print("Gradient check passed:", dummy.grad is not None)
