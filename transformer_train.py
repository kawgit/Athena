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

from torch.utils.data import DataLoader
import torch

from checkpoint import load_checkpoint
from dataset import load_dataset
from device import device
from settings import *
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from trainer import Trainer
from transformer import Transformer

transformer = Transformer().to(device)

def criterion(output_probs, expected_indexes):
    return cross_entropy(output_probs.reshape(-1, vocab_size), expected_indexes.reshape(-1))

optimizer = AdamW(transformer.parameters(), lr=1, fused=True)
scheduler = LambdaLR(optimizer, lambdalr)

trainer = Trainer(criterion, optimizer, scheduler)

load_checkpoint(transformer, trainer)

dataloader = DataLoader(load_dataset(), batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
trainer.fit(transformer, dataloader, 10)
