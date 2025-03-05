import torch
from transformers import AutoModelForCausalLM
from settings import hfmodel_name
from hfmodel import load_hfmodel
import time

start_time = time.time()

model = load_hfmodel()
total_size = 0

for name, param in model.named_parameters():
    size = torch.prod(torch.tensor(param.shape)).item()
    print(f"{name:<50} {str(param.shape):<30} {size}")
    total_size += size

print(f"Total param count {total_size}")
print(f"Time elapsed {time.time() - start_time}")