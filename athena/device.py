import torch

cpu = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

torch.set_float32_matmul_precision('high')

print('device:', device)