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

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from device import device
from settings import *

class Athena(nn.Module):

    def __init__(self):
        super().__init__()

        self.register_buffer(
            "cos_buffer",
            torch.tensor(np.array([
                [
                    math.cos(m * 10000 ** ((-2 / key_size) * (i % (key_size // 2)))) for i in range(key_size)
                ] for m in range(context_size)
            ])).float()
        )

        self.register_buffer(
            "sin_buffer",
            torch.tensor(np.array([
                [
                    math.sin(m * 10000 ** ((-2 / key_size) * (i % (key_size // 2)))) * (-1 if i < key_size // 2 else 1) for i in range(key_size)
                ] for m in range(context_size)
            ])).float()
        )

        self.table = nn.Embedding(vocab_size, embedding_size)
        self.sas = nn.ModuleList([AthenaSA() for i in range(num_layers)])
        self.ffns = nn.ModuleList([AthenaFFN() for i in range(num_layers)])
        self.norm = nn.RMSNorm(embedding_size, eps=1e-05)
        self.predictor = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, tokens):

        current_context_size = tokens.shape[1]
        assert(current_context_size <= context_size)

        cos_buffer = self.cos_buffer[:current_context_size, :]
        sin_buffer = self.sin_buffer[:current_context_size, :]

        embeddings = self.table(tokens)
    
        for sa, ffn in zip(self.sas, self.ffns):
            embeddings = sa(embeddings, cos_buffer, sin_buffer)
            embeddings = ffn(embeddings)
        
        embeddings = self.norm(embeddings)

        logits = self.predictor(embeddings if self.training else embeddings[:, -1, :])

        return logits if self.training else functional.softmax(logits, dim=-1).cpu().detach()

    def generate(self, seed_tokens, num_new_tokens, end_token_ids=[]):

        self.eval()

        text_tokens = seed_tokens.copy()

        for i in range(num_new_tokens):

            input = torch.tensor(text_tokens[-context_size:]).to(next(self.parameters()).device).reshape(1, -1)
            output = self.forward(input).reshape(-1).detach().numpy()

            new_token = np.random.choice(range(vocab_size), p=output)
            text_tokens.append(new_token)

            yield new_token

            if new_token in end_token_ids:
                break


class AthenaSA(nn.Module):
    def __init__(self):

        super().__init__()

        self.norm = nn.RMSNorm(embedding_size, eps=1e-05)
        self.qkv_proj = nn.Linear(embedding_size, 2 * key_size * num_heads + head_size * num_heads, bias=False)
        self.out_proj = nn.Linear(num_heads * head_size, embedding_size, bias=False)

    def forward(self, embeddings, cos_buffer, sin_buffer):

        batch_size, current_context_size, _ = embeddings.shape

        normed = self.norm(embeddings) # B, C, E
        qkv = self.qkv_proj(normed) # B, C, Q * H + K * H + V * H

        keys_start = key_size * num_heads
        values_start = 2 * keys_start

        queries = qkv[..., :keys_start].reshape((batch_size, current_context_size, num_heads, key_size)).transpose(1, 2) # B, H, C, Q
        keys = qkv[..., keys_start:values_start].reshape((batch_size, current_context_size, num_heads, key_size)).transpose(1, 2) # B, H, C, K
        values = qkv[..., values_start:].reshape((batch_size, current_context_size, num_heads, head_size)).transpose(1, 2) # B, H, C, V

        queries = self.rotate(queries, cos_buffer, sin_buffer)
        keys = self.rotate(keys, cos_buffer, sin_buffer)

        outputs = functional.scaled_dot_product_attention(
            queries.contiguous(),
            keys.contiguous(),
            values.contiguous(),
            is_causal=True,
            dropout_p=(0 if not self.training else 0.1)
        ) # B, H, C, V
        outputs = outputs.transpose(1, 2).contiguous() # B, C, H, V
        outputs = outputs.reshape((batch_size, current_context_size, num_heads * head_size)) # B, C, H * V
        outputs = self.out_proj(outputs) # B, C, E

        return embeddings + outputs

    def rotate(self, features, cos_buffer, sin_buffer):
        
        swapped = torch.cat((features[..., key_size // 2:], features[..., :key_size // 2]), dim=-1)

        return cos_buffer * features + sin_buffer * swapped

class AthenaFFN(nn.Module):
    def __init__(self):

        super().__init__()

        self.norm = nn.RMSNorm(embedding_size, eps=1e-05)
        self.up_proj = nn.Linear(embedding_size, 2 * hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, embedding_size, bias=False)
    
    def forward(self, embeddings):

        normed = self.norm(embeddings)
        gate_up = self.up_proj(normed)

        gate = gate_up[..., :hidden_size]
        up = gate_up[..., hidden_size:]

        down = self.down_proj(up * functional.silu(gate))

        return embeddings + down

def load_athena():

    athena = Athena()

    if resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")

        athena.load_state_dict(torch.load(checkpoint_path, weights_only=True)["model"])
    
    else:
        print(f"Loading base from base.pt")

        athena.load_state_dict(torch.load("base.pt", weights_only=True))


    return athena
