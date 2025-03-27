#    Copyright 2025 Kenneth Wilber (kawgit)

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from settings import *
import torch.utils._device

def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors="replace") as file:
        return file.read()

def format_number(number, total_length=7):
    return f"{number:0{total_length}.6f}"

def split_into_segments(text):
    
    segments = []
    
    start = 0

    for i, (a, b) in enumerate(zip(text[:-1], text[1:])):

        if i == start:
            continue

        if a.isspace() and not b.isspace():
            segments.append(text[start:i])
            start = i

        if a.isalnum() and not b.isalnum() and not b == "'" and not b == "’":
            segments.append(text[start:i+1])
            start = i + 1

    else:
        segments.append(text[start:])

    return segments

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