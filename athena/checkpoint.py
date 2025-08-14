import torch
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR

from athena.model import Athena
from athena.utils import EmptyInitOnDevice
from athena.device import device

def get_checkpoint_path(name):
    return f"checkpoints/{name}.ckpt"

def save_checkpoint(athena, optimizer=None, scheduler=None):
    
    checkpoint = {
        'config': athena.config,
        'weights': athena.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None
    }

    path = get_checkpoint_path(athena.name)
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))

    if os.path.exists(path):
        os.rename(path, path + ".old")
    
    torch.save(checkpoint, path)

def load_checkpoint(name):
    
    path = get_checkpoint_path(name)
    checkpoint = torch.load(path, weights_only=False)
    
    with EmptyInitOnDevice(device):
        athena = Athena(checkpoint["config"]).to(device)

    athena.load_state_dict(checkpoint["weights"], strict=False)

    optimizer = AdamW(athena.parameters(), lr=3e-5)
    if checkpoint['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=50)
    main_sched = ConstantLR(optimizer, factor=1.0, total_iters=950)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, main_sched], milestones=[50])
    
    if checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
        
    return athena, optimizer, scheduler