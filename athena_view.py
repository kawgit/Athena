import torch

from athena.checkpoint import load_checkpoint
from settings import checkpoint_path

model, _ = load_checkpoint(checkpoint_path)

total_size = 0

for name, param in model.named_parameters():
    size = torch.prod(torch.tensor(param.shape)).item()
    print(f"{name:<50} {str(param.shape):<30} {size}")
    total_size += size

print(f"Total param count {total_size}")
