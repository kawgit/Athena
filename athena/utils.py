from bisect import bisect_left
from typing import Callable, Tuple, Type
import hashlib
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.utils._device

class EmptyInitOnDevice(torch.overrides.TorchFunctionMode):
    def __init__(self, device=None):
        self.device = device

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if getattr(func, '__module__', None) == 'torch.nn.init':
            if 'tensor' in kwargs:
                return kwargs['tensor']
            else:
                return args[0]
        if self.device is not None and func in torch.utils._device._device_constructors() and kwargs.get('device') is None:
            kwargs['device'] = self.device
        return func(*args, **kwargs)

class Timer:
    def __init__(self, title=None):
        self.title = title
        self.start = time.time()
        self.elapsed = None

    def __enter__(self):
        if self.title != None:
            print(f"{self.title}... ", end='', flush=True)
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = time.time() - self.start
        if self.title != None:
            print(f"Done in {self.elapsed:.2f} seconds.")

    def __float__(self):
        return self.elapsed if self.elapsed is not None else time.time() - self.start
    
class Throttle:
    def __init__(self, title, interval):
        self.title = title
        self.interval = interval
        self._last_run = time.time()
        self._start = None

    def __enter__(self):
        now = time.time()
        if now - self._last_run >= self.interval:
            self._last_run = now
            self._start = now
            if self.title != None:
                print(f"{self.title}... ", end="", flush=True)
            return True
        return False

    def __exit__(self, exc_type, exc_value, traceback):
        if self._start is not None:
            elapsed = time.time() - self._start
            if self.title != None:
                print(f"Done in {elapsed:.2f} seconds.")
            self._start = None
        return False

def generate_model_name(config):
    return config["name"] if config["name"] != None else f"athena_{"_".join(str(value) for value in config.values() if type(value) in [int, float, list])}"

def save_tensor_as_image(tensor, filename="output.png", cmap="viridis"):
    """
    Save a 2D tensor (or numpy array) as an image.

    Args:
        tensor: 2D torch.Tensor or np.ndarray of floats
        filename: Path to save the image (e.g. 'image.png')
        cmap: Matplotlib colormap (default: 'viridis')
    """
    # Convert PyTorch tensor to numpy if needed
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    # Ensure 2D
    if tensor.ndim != 2:
        raise ValueError("Input must be a 2D tensor or array")
    
    # Normalize values to [0,1] for display
    norm_tensor = (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor) + 1e-8)
    
    plt.imsave(filename, norm_tensor, cmap=cmap)
    print(f"Image saved to {filename}")

def linear_interpolator(*points: Tuple[float, float]) -> Callable[[float], float]:
    """
    Given (x, y) points as separate arguments, return a function f(x) that
    linearly interpolates between them.
    
    Example:
        f = linear_interpolator((0,0), (1,2), (2,4))
        f(0.5) -> 1.0
    """
    if not points:
        raise ValueError("At least one point must be provided")
        
    # Sort by x just in case
    points = sorted(points, key=lambda p: p[0])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    def f(x: float) -> float:
        # Clamp at endpoints
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        
        # Find interval
        idx = bisect_left(xs, x)
        x0, y0 = xs[idx - 1], ys[idx - 1]
        x1, y1 = xs[idx], ys[idx]
        
        # Linear interpolation
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    return f

def scale_interpolator(interpolator, scale):
    return lambda epoch: scale * interpolator(epoch)

def ith_element(t: torch.Tensor, i: int):
    # bounds check (optional)
    if not (0 <= i < t.numel()):
        raise IndexError("i out of range")

    if t.is_contiguous():
        # O(1): just a view, no copy
        return t.view(-1)[i]
    else:
        # O(rank): compute the multi-dim index, no full flatten/copy
        idx = torch.unravel_index(torch.tensor(i, device=t.device), t.shape)
        return t[idx]

@torch.no_grad()
def add_or_create(param, attr, value):
    setattr(param, attr, value if not hasattr(param, attr) else getattr(param, attr) + value)
    
@torch.no_grad()
def lerp_or_create(param, attr, value, momentum=.9):
    setattr(param, attr, value if not hasattr(param, attr) else momentum * getattr(param, attr) + (1 - momentum) * value)
    
@torch.no_grad()
def concat_or_create(param, attr, value):
    setattr(param, attr, value if not hasattr(param, attr) else torch.concat((getattr(param, attr), value)))
    
def seed_all(seed: int = 42, deterministic: bool = True):
    """
    Seed all randomness sources for full reproducibility.
    """
    # --- Python ---
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # --- NumPy ---
    np.random.seed(seed)

    # --- PyTorch ---
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # --- CuDNN backend determinism ---
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # --- PyTorch 2.0+ Deterministic Algorithms ---
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"[Seeded everything with seed={seed}]")
    
def str_to_seed(s: str, bits: int = 64) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % (2**bits)

def sample_indices_torch(n: int, k: int, seed_str: str, device: str | None = None) -> torch.Tensor:
    if not (0 <= k <= n):
        raise ValueError("k must be in [0, n].")
    # Isolated generator; does not affect global torch RNG state
    g = torch.Generator(device="cpu")
    g.manual_seed(str_to_seed(seed_str))
    idx = torch.randperm(n, generator=g)[:k]           # on CPU, deterministic from seed_str
    return idx.to(device) if device is not None else idx
