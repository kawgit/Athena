import torch
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

    path = get_checkpoint_path(athena.config.name)
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))

    if os.path.exists(path):
        os.rename(path, path + ".old")
    
    torch.save(checkpoint, path)

def load_checkpoint(name, config_modifier=None):
    
    path = get_checkpoint_path(name)
    checkpoint = torch.load(path, weights_only=False)
    
    if config_modifier is not None:
        config_modifier(checkpoint["config"])
        athena = Athena(checkpoint["config"])
    else:
        with EmptyInitOnDevice(device):
            athena = Athena(checkpoint["config"])

    athena.load_state_dict(checkpoint["weights"], strict=config_modifier is None)

    optimizer = AdamW(athena.parameters(), lr=3e-5)
    if checkpoint['optimizer'] is not None and config_modifier is None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    if checkpoint['scheduler'] is not None and config_modifier is None:
        scheduler.load_state_dict(checkpoint['scheduler'])
        
    return athena, optimizer, scheduler