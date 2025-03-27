import time
import torch

from athena import Athena
from utils import EmptyInitOnDevice

start_time = time.time()

with EmptyInitOnDevice('cpu'):
    model = Athena()

total_size = 0

for name, param in model.named_parameters():
    size = torch.prod(torch.tensor(param.shape)).item()
    print(f"{name:<50} {str(param.shape):<30} {size}")
    total_size += size

print(f"Total param count {total_size}")
