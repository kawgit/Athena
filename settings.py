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

import os
import math

hfmodel_name = "microsoft/Phi-3-mini-4k-instruct"

dataset_name = "harrypotter1"

vocab_size = 32064

context_size = 4096
key_size = 96
head_size = 96
num_heads = 32
embedding_size = 3072
hidden_size = 8192
num_layers = 32

assert(key_size % 2 == 0)

batch_size = 16

lr_init = 3e-5
lr_max = 3e-4
lr_min = 3e-6
warmup_steps = 600 
period_steps = 49 * warmup_steps
resume = True

def lambdalr(step):
    if step < warmup_steps:
        return lr_init + (lr_max - lr_init) * step / warmup_steps
    if step < warmup_steps + period_steps:
        return lr_min + (lr_max - lr_min) * math.cos((step - warmup_steps) * math.pi / 2 / period_steps) ** 4
    return lr_min

model_name = f"{dataset_name}_c{context_size}_m{embedding_size}_k{key_size}_h{head_size}_l{num_layers}_v{hidden_size}"
checkpoint_path = os.path.join('checkpoints', model_name + '.checkpoint')
tokenizer_path = f"tokenizers/{dataset_name}.pickle"
