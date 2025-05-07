import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from athena.device import device
from athena.generation import AthenaGeneration

class Athena(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config

        self.register_buffer(
            "causal_buffer",
            torch.ones(self.config.context_size, self.config.context_size, dtype=torch.bool).tril(diagonal=0)
        )

        self.register_buffer(
            "cos_buffer",
            torch.tensor(np.array([
                [
                    math.cos(m * 10000 ** ((-2 / self.config.key_size) * (i % (self.config.key_size // 2)))) for i in range(self.config.key_size)
                ] for m in range(self.config.context_size)
            ], dtype=np.float32)).float()
        )

        self.register_buffer(
            "sin_buffer",
            torch.tensor(np.array([
                [
                    math.sin(m * 10000 ** ((-2 / self.config.key_size) * (i % (self.config.key_size // 2)))) * (-1 if i < self.config.key_size // 2 else 1) for i in range(self.config.key_size)
                ] for m in range(self.config.context_size)
            ], dtype=np.float32)).float()
        )
        
        self.table = nn.Embedding(self.config.vocab_size, self.config.embedding_size)
        self.sas = nn.ModuleList([AthenaSA(self.config) for i in range(self.config.num_layers)])
        self.ffns = nn.ModuleList([AthenaFFN(self.config) for i in range(self.config.num_layers)])
        self.norm = nn.RMSNorm(self.config.embedding_size, eps=1e-05)
        self.vocab_proj = nn.Linear(self.config.embedding_size, self.config.vocab_size, bias=False)
        self.quality_proj = nn.Linear(self.config.embedding_size, 1, bias=False)

    def forward(self, tokens, past_kvs=None, return_logits=True, return_qualities=False):
        
        self.table.weight.data[0] = torch.zeros(self.config.embedding_size)

        use_cache = past_kvs != None and past_kvs != "init"
        create_cache = past_kvs != None

        batch_size, queries_length = tokens.shape
        keys_length = queries_length + (past_kvs[0].shape[3] if use_cache else 0)
        queries_start = keys_length - queries_length
        
        causal_buffer = self.causal_buffer[queries_start:keys_length, :keys_length]
        cos_buffer = (self.cos_buffer[queries_start:keys_length, :], self.cos_buffer[:keys_length, :])
        sin_buffer = (self.sin_buffer[queries_start:keys_length, :], self.sin_buffer[:keys_length, :])

        present_kvs = []
        embeddings = self.table(tokens)
    
        for sa, ffn, past_kv in zip(self.sas, self.ffns, zip(*past_kvs) if use_cache else [None] * len(self.sas)):
            embeddings, present_kv = sa(embeddings, cos_buffer, sin_buffer, causal_buffer, past_kv=past_kv)
            embeddings = ffn(embeddings)

            if create_cache:
                present_kvs.append(present_kv)
        
        embeddings = self.norm(embeddings)
        
        if create_cache:
            present_ks = torch.stack([present_k for present_k, _ in present_kvs])
            present_vs = torch.stack([present_v for _, present_v in present_kvs])
            present_kvs = (present_ks, present_vs)
        
        result = {}
        
        if return_logits:
            result["logits"] = self.vocab_proj(embeddings)
            
        if return_qualities:
            result["qualities"] = self.quality_proj(embeddings).squeeze(-1)
        
        if create_cache:
            result["present_kvs"] = present_kvs

        return result

    def generate(self, seed_tokens, max_new_tokens, stream=False, athena_compiled=None, temperature=1.0):

        generation = AthenaGeneration(self, athena_compiled, seed_tokens, temperature=temperature)

        if stream:
            return self.generate_internal(generation, max_new_tokens, stream)
        else:
            return next(self.generate_internal(generation, max_new_tokens, stream))

    def generate_internal(self, generation, max_new_tokens, stream):

        for i in range(max_new_tokens):
            
            generation.step()

            if stream:
                yield generation
        
        if not stream:
            yield generation

class AthenaSA(nn.Module):
    def __init__(self, config):

        super().__init__()

        self.config = config

        self.norm = nn.RMSNorm(self.config.embedding_size, eps=1e-05)
        self.qkv_proj = nn.Linear(self.config.embedding_size, 2 * self.config.key_size * self.config.num_heads + self.config.head_size * self.config.num_heads, bias=False)
        self.out_proj = nn.Linear(self.config.num_heads * self.config.head_size, self.config.embedding_size, bias=False)

    def forward(self, embeddings, cos_buffer, sin_buffer, causal_buffer, past_kv=None):

        normed = self.norm(embeddings) # B, C, E
        qkv = self.qkv_proj(normed) # B, C, H * Q + H * K + H * V

        keys_start = self.config.key_size * self.config.num_heads
        values_start = 2 * keys_start

        queries = qkv[..., :keys_start] # B, C, H * Q
        keys = qkv[..., keys_start:values_start] # B, C, H * K
        values = qkv[..., values_start:] # B, C, H * V
        
        batch_size, current_context_size, _ = embeddings.shape
        qk_shape = (batch_size, current_context_size, self.config.num_heads, self.config.key_size)
        v_shape = (batch_size, current_context_size, self.config.num_heads, self.config.head_size)

        queries = queries.reshape(qk_shape).transpose(1, 2) # B, H, C, Q
        keys = keys.reshape(qk_shape).transpose(1, 2) # B, H, C, K
        values = values.reshape(v_shape).transpose(1, 2) # B, H, C, V

        if past_kv != None:
            past_k, past_v = past_kv
            keys = torch.cat((past_k, keys), dim=-2)
            values = torch.cat((past_v, values), dim=-2)

        present_kv = (keys.detach(), values.detach())

        queries = self.rotate(queries, cos_buffer[0], sin_buffer[0])
        keys = self.rotate(keys, cos_buffer[1], sin_buffer[1])

        queries = queries.contiguous()
        keys = keys.contiguous()
        values = values.contiguous()
        
        if device.type == "mps" and self.config.key_size != self.config.head_size:
            attn = queries @ keys.transpose(-2, -1) / math.sqrt(self.config.key_size)
            attn = attn.masked_fill(causal_buffer == False, float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            outputs = attn @ values
        else:
            outputs = functional.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=causal_buffer
            ) # B, H, C, V
    
        outputs = outputs.transpose(1, 2).contiguous() # B, C, H, V
        outputs = outputs.flatten(-2) # B, C, H * V
        outputs = self.out_proj(outputs) # B, C, E

        return embeddings + outputs, present_kv

    def rotate(self, features, cos_buffer, sin_buffer):
        
        swapped = torch.cat((features[..., self.config.key_size // 2:], features[..., :self.config.key_size // 2]), dim=-1)
        
        return cos_buffer * features + sin_buffer * swapped

class AthenaFFN(nn.Module):
    def __init__(self, config):

        super().__init__()

        self.config = config

        self.norm = nn.RMSNorm(self.config.embedding_size, eps=1e-05)
        self.up_proj = nn.Linear(self.config.embedding_size, 2 * self.config.hidden_size, bias=False)
        self.down_proj = nn.Linear(self.config.hidden_size, self.config.embedding_size, bias=False)
    
    def forward(self, embeddings):

        normed = self.norm(embeddings)
        gate_up = self.up_proj(normed)

        gate = gate_up[..., :self.config.hidden_size]
        up = gate_up[..., self.config.hidden_size:]

        down = self.down_proj(up * functional.silu(gate))

        return embeddings + down

class AthenaCompiled(nn.Module):
    def __init__(self, athena):
        super().__init__()
        self.athena = athena
        self.athena_compiled = torch.compile(athena) if device.type == "cuda" else athena
        self.config = athena.config
    
    def forward(self, *args, **kwargs):
        return self.athena_compiled(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        return self.athena.generate(*args, **kwargs, athena_compiled=self.athena_compiled)
    
    def train(self):
        self.athena.train()
        self.athena_compiled.train()
    
    def eval(self):
        self.athena.eval()
        self.athena_compiled.eval()