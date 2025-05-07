import time
import torch
import torch.nn.functional as F
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

def make_chat_pretty(chat):

    return chat.replace("<|pad|>", "").replace("<|input|>", "\n\nInput:\n").replace("<|output|>", "\n\nOutput:\n")

def print_graph(gfn, indent=0):
    if indent > 5:
        return

    print(" " * indent + str(gfn))
    
    if hasattr(gfn, 'next_functions'):
        for fn, _ in gfn.next_functions:
            if fn is not None:
                print_graph(fn, indent + 2)

def left_pad_and_stack(tensor_list):
    max_len = max(len(t) for t in tensor_list)
    padded = [F.pad(torch.tensor(t, dtype=torch.long) if type(t) == list else t, (max_len - len(t), 0)) for t in tensor_list]
    return torch.stack(padded)

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
