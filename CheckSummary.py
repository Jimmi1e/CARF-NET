import torch
import torch.nn as nn
import torch.nn.functional as F
from models import PatchmatchNet
from torchinfo import summary
class PatchmatchSummaryWrapper(nn.Module):
    def __init__(self, patchmatch_net):
        super().__init__()
        self.patchmatch_net = patchmatch_net

    def forward(self, x, **kwargs):

        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        images = [x, x]
        intrinsics = torch.rand(x.size(0), 2, 3, 3, device=x.device)
        extrinsics = torch.rand(x.size(0), 2, 4, 4, device=x.device)
        depth_min = torch.tensor([0.1], device=x.device)
        depth_max = torch.tensor([10.0], device=x.device)
        
        output = self.patchmatch_net(images, intrinsics, extrinsics, depth_min, depth_max)
        return output[0]



patchmatch_net = PatchmatchNet(
    patchmatch_interval_scale=[0.005, 0.0125, 0.025],
    propagation_range=[6, 4, 2],
    patchmatch_iteration=[1, 2, 2],
    patchmatch_num_sample=[8, 8, 16],
    propagate_neighbors=[0, 8, 16],
    evaluate_neighbors=[9, 9, 9],
    Attention_Selection='CBAM',
)


model = PatchmatchSummaryWrapper(patchmatch_net)
summary(model, input_size=(3, 512, 512), batch_size=1, device="cpu")
