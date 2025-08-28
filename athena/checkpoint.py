from torch import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR
import os
import torch

from athena.device import device
from athena.model import Athena
from athena.utils import EmptyInitOnDevice

def get_checkpoint_path(name):
    return f"checkpoints/{name.removesuffix('.ckpt')}.ckpt"

def save_checkpoint(athena, optimizer=None, scheduler=None, scaler=None):
    checkpoint = {
        'config': athena.config,
        'weights': athena.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'scaler': scaler.state_dict() if scaler is not None else None,
    }
    path = get_checkpoint_path(athena.name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.replace(path, path + ".old")
    torch.save(checkpoint, path)


def select_amp(dev: torch.device):
    if dev.type == "cuda" and not torch.cuda.is_bf16_supported():
        return torch.float16, True
    return torch.bfloat16, False


def load_checkpoint(name):
    path = get_checkpoint_path(name)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    with EmptyInitOnDevice(device):
        athena = Athena(checkpoint['config']).to(device)

    missing, unexpected = athena.load_state_dict(checkpoint['weights'], strict=False)
    if missing or unexpected:
        print(f"[WARN] Missing keys: {missing}\n[WARN] Unexpected keys: {unexpected}")

    optimizer = AdamW(athena.parameters(), lr=3e-5)
    if checkpoint.get('optimizer') is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler = ConstantLR(optimizer, factor=1.0)
    if checkpoint.get('scheduler') is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    amp_dtype, use_scaler = select_amp(device)
    scaler = GradScaler(enabled=use_scaler)
    if use_scaler and checkpoint.get('scaler') is not None:
        scaler.load_state_dict(checkpoint['scaler'])

    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype)

    return athena, optimizer, scheduler, scaler, autocast_ctx
