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
from athena_tokenizer import padding_token_id, end_token_id

class Athena(nn.Module):

    def __init__(self):
        super().__init__()

        self.register_buffer(
            "causal_buffer",
            torch.ones(context_size, context_size, dtype=torch.bool).tril(diagonal=0)
        )

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

    def forward(self, tokens, past_kvs=None):

        use_cache = type(past_kvs) == list
        create_cache = use_cache or past_kvs == "init"

        batch_size, queries_length = tokens.shape
        keys_length = queries_length + (past_kvs[0][0].shape[2] if use_cache else 0)
        queries_start = keys_length - queries_length
        
        causal_buffer = self.causal_buffer[queries_start : keys_length, :keys_length]
        cos_buffer = (self.cos_buffer[queries_start : keys_length, :], self.cos_buffer[:keys_length, :])
        sin_buffer = (self.sin_buffer[queries_start : keys_length, :], self.sin_buffer[:keys_length, :])

        present_kvs = []
        embeddings = self.table(tokens)
    
        for sa, ffn, past_kv in zip(self.sas, self.ffns, past_kvs if use_cache else [None] * len(self.sas)):
            embeddings, present_kv = sa(embeddings, cos_buffer, sin_buffer, causal_buffer, past_kv=past_kv)
            embeddings = ffn(embeddings)

            if create_cache:
                present_kvs.append(present_kv)
        
        embeddings = self.norm(embeddings)
        logits = self.predictor(embeddings)

        return (logits, present_kvs) if create_cache else logits

    def generate(self, seed_tokens, max_new_tokens, reply_queue=None):

        if reply_queue == None:
            reply_queue = [] * len(seed_tokens)

        cum_tokens = seed_tokens
        new_tokens = cum_tokens
        past_kvs = "init"
        active_indexes = list(range(len(seed_tokens)))
        replying = [False] * len(seed_tokens)

        for i in range(max_new_tokens):

            inputs = torch.tensor(new_tokens)[:, -context_size:].to(next(self.parameters()).device)
            logits, past_kvs = self.forward(inputs, past_kvs=past_kvs)
            probs = functional.softmax(logits[:, -1, :], dim=-1).detach().cpu().numpy()

            new_tokens = [[np.random.choice(len(prob), p=prob)] for prob in probs]

            for new_index, old_index in reversed(list(enumerate(active_indexes))):

                if replying[old_index]:

                    if len(reply_queue[old_index][0]) == 0:
                        replying[old_index] = False
                        del reply_queue[old_index][0]

                    else:
                        new_tokens[new_index][0] = reply_queue[old_index][0][0]
                        del reply_queue[old_index][0][0]

                cum_tokens[old_index].append(new_tokens[new_index][0])

                if not replying[old_index] and new_tokens[new_index][0] == end_token_id:

                    if len(reply_queue[old_index]) != 0:

                        replying[old_index] = True

                    else:

                        del active_indexes[new_index]
                        del new_tokens[new_index]
                        
                        for i, (past_k, past_v) in enumerate(past_kvs):

                            mask = torch.ones(past_k.shape[0], dtype=torch.bool)
                            mask[new_index] = False
                            
                            past_k = past_k[mask]
                            past_v = past_v[mask]

                            past_kvs[i] = (past_k, past_v)
            
            yield cum_tokens

            if len(active_indexes) == 0:
                break

class AthenaSA(nn.Module):
    def __init__(self):

        super().__init__()

        self.norm = nn.RMSNorm(embedding_size, eps=1e-05)
        self.qkv_proj = nn.Linear(embedding_size, 2 * key_size * num_heads + head_size * num_heads, bias=False)
        self.out_proj = nn.Linear(num_heads * head_size, embedding_size, bias=False)

    def forward(self, embeddings, cos_buffer, sin_buffer, causal_buffer, past_kv=None):

        normed = self.norm(embeddings) # B, C, E
        qkv = self.qkv_proj(normed) # B, C, H * Q + H * K + H * V

        keys_start = key_size * num_heads
        values_start = 2 * keys_start

        queries = qkv[..., :keys_start] # B, C, H * Q
        keys = qkv[..., keys_start:values_start] # B, C, H * K
        values = qkv[..., values_start:] # B, C, H * V

        batch_size, current_context_size, _ = embeddings.shape
        qk_shape = (batch_size, current_context_size, num_heads, key_size)
        v_shape = (batch_size, current_context_size, num_heads, head_size)

        queries = queries.reshape(qk_shape).transpose(1, 2) # B, H, C, Q
        keys = keys.reshape(qk_shape).transpose(1, 2) # B, H, C, K
        values = values.reshape(v_shape).transpose(1, 2) # B, H, C, V

        if past_kv != None:
            past_keys, past_values = past_kv
            keys = torch.cat((past_keys, keys), dim=-2)
            values = torch.cat((past_values, values), dim=-2)

        present_kv = (keys.detach(), values.detach())

        queries = self.rotate(queries, cos_buffer[0], sin_buffer[0])
        keys = self.rotate(keys, cos_buffer[1], sin_buffer[1])

        outputs = functional.scaled_dot_product_attention(
            queries.contiguous(),
            keys.contiguous(),
            values.contiguous(),
            attn_mask=causal_buffer,
            dropout_p=(0 if not self.training else 0.1)
        ) # B, H, C, V

        outputs = outputs.transpose(1, 2).contiguous() # B, C, H, V
        outputs = outputs.flatten(-2) # B, C, H * V
        outputs = self.out_proj(outputs) # B, C, E

        return embeddings + outputs, present_kv

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
        athena.load_state_dict(torch.load("base.pt", weights_only=True), strict=False)


    return athena
 