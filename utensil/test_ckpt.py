import torch

ckpt_path = r'F:\Canada\Concordia\COEN691O\Project\Traininglog\model1_1_000007.ckpt'
# ckpt_path = r'F:\Canada\Concordia\COEN691O\Project\PatchmatchNet\checkpoints\params_000007.ckpt'
checkpoint = torch.load(ckpt_path)

model = checkpoint['model']
for name, param in model.items():
    print(name, param.size())