import math
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from athena.attention import attention
from athena.device import device
from athena.generation import AthenaGeneration

class Athena(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.__dict__.update(config)
        
        assert(self.num_heads % self.num_kv_heads == 0)
        assert(not hasattr(self, "window_sizes") or len(self.window_sizes) == self.num_layers)
        
        self.register_buffer(
            "cos_buffer",
            torch.zeros((self.context_size, self.key_size), dtype=torch.bfloat16),
            persistent=False
        )
        
        self.register_buffer(
            "sin_buffer",
            torch.zeros((self.context_size, self.key_size), dtype=torch.bfloat16),
            persistent=False
        )
        
        i = torch.arange(self.key_size)
        half = self.key_size // 2
        offset = i % half

        exponent = (-2.0 / self.key_size) * offset
        scales = torch.pow(10000, exponent)

        m = torch.arange(self.context_size)
        angles = m[:, None] * scales[None, :]
        
        self.cos_buffer = torch.cos(angles)
        self.sin_buffer = torch.sin(angles)
        self.sin_buffer[:, :half] *= -1
        
        self.table = nn.Embedding(self.vocab_size, self.embedding_size)
        self.sas = nn.ModuleList([AthenaSA(i, self.config) for i in range(self.num_layers)])
        self.ffns = nn.ModuleList([AthenaFFN(i, self.config) for i in range(self.num_layers)])
        self.norm = nn.RMSNorm(self.embedding_size, eps=1e-05, elementwise_affine=False)
        self.vocab_proj = nn.Linear(self.embedding_size, self.vocab_size, bias=False)

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.config[key] = value
            
        for sa in self.sas:
            sa.update_config(**kwargs)
            
        for ffn in self.ffns:
            ffn.update_config(**kwargs)

    def forward(self, tokens, past_kvs=None, return_logits=True):
        
        self.table.weight.data[0] = torch.zeros(self.embedding_size)

        use_cache = past_kvs != None and past_kvs != "init"
        create_cache = past_kvs != None

        batch_size, q_len = tokens.shape
        q_start = past_kvs[0].shape[3] if use_cache else 0
        k_len = q_start + q_len
        
        cos_buffer = (self.cos_buffer[q_start:k_len], self.cos_buffer[:k_len])
        sin_buffer = (self.sin_buffer[q_start:k_len], self.sin_buffer[:k_len])
        
        present_kvs = []
        embeddings = self.table(tokens)
        past_kvs = zip(*past_kvs) if use_cache else [None] * len(self.sas)
        
        for sa, ffn, past_kv in zip(self.sas, self.ffns, past_kvs):

            embeddings, present_kv = sa(embeddings, cos_buffer, sin_buffer, past_kv=past_kv)
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
    def __init__(self, layer_index, config):

        super().__init__()

        self.config = config
        self.__dict__.update(config)
        
        self.window_size = config["window_sizes"][layer_index]

        self.norm = nn.RMSNorm(self.embedding_size, eps=1e-05, elementwise_affine=False)
        self.queries_proj = nn.Linear(self.embedding_size, self.key_size * self.num_heads, bias=False)
        self.keys_proj = nn.Linear(self.embedding_size, self.key_size * self.num_kv_heads, bias=False)
        self.values_proj = nn.Linear(self.embedding_size, self.head_size * self.num_kv_heads, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_size, self.embedding_size, bias=False)

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.config[key] = value

    def forward(self, embeddings: Tensor, cos_buffer: Tensor, sin_buffer: Tensor, past_kv=None):
        
        normed: Tensor = self.norm(embeddings) # B, C, E

        queries: Tensor = self.queries_proj(normed) # B, C, H_q * Q
        keys: Tensor = self.keys_proj(normed) # B, C, H_kv * K
        values: Tensor = self.values_proj(normed) # B, C, H_kv * V
        
        batch_size, current_context_size, _ = embeddings.shape
        q_shape = (batch_size, current_context_size, self.num_heads, self.key_size)
        k_shape = (batch_size, current_context_size, self.num_kv_heads, self.key_size)
        v_shape = (batch_size, current_context_size, self.num_kv_heads, self.head_size)

        queries = queries.reshape(q_shape).transpose(1, 2) # B, H_q, C, Q
        keys = keys.reshape(k_shape).transpose(1, 2) # B, H_kv, C, K
        values = values.reshape(v_shape).transpose(1, 2) # B, H_kv, C, V

        if past_kv != None:
            past_k, past_v = past_kv
            keys = torch.cat((past_k, keys), dim=-2)
            values = torch.cat((past_v, values), dim=-2)
        
        present_kv = (keys.detach(), values.detach())

        queries = self.rotate(queries, cos_buffer[0], sin_buffer[0])
        keys = self.rotate(keys, cos_buffer[1], sin_buffer[1])
        
        outputs = attention(queries, keys, values, window_size=self.window_size)
        outputs = outputs.transpose(1, 2).contiguous() # B, C, H_q, V
        outputs = outputs.flatten(-2) # B, C, H_q * V
        outputs = self.out_proj(outputs) # B, C, E

        return embeddings + outputs, present_kv

    def rotate(self, features, cos_buffer, sin_buffer):
        
        swapped = torch.cat((features[..., self.key_size // 2:], features[..., :self.key_size // 2]), dim=-1)
        
        return cos_buffer * features + sin_buffer * swapped

class AthenaFFN(nn.Module):
    def __init__(self, layer_index, config):

        super().__init__()

        self.config = config
        self.__dict__.update(config)
        
        if self.hidden_size <= 0:
            return

        self.norm = nn.RMSNorm(self.embedding_size, eps=1e-05, elementwise_affine=False)
        self.up_proj = nn.Linear(self.embedding_size, 2 * self.hidden_size, bias=False)
        self.down_proj = nn.Linear(self.hidden_size, self.embedding_size, bias=False)
    
    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.config[key] = value
    
    def forward(self, embeddings):

        if self.hidden_size <= 0:
            return embeddings
        
        normed = self.norm(embeddings)
        gate_up = self.up_proj(normed)

        gate = gate_up[..., :self.hidden_size]
        up = gate_up[..., self.hidden_size:]

        down = self.down_proj(up * functional.silu(gate))

        return embeddings + down

class AthenaCompiled(nn.Module):
    def __init__(self, athena):
        super().__init__()
        self.athena = athena
        self.athena_compiled = torch.compile(athena) if device.type == "cuda" else athena
        self.config = athena.config
        
        self.__dict__.update(self.config)
    
    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.config[key] = value
            
        self.athena.update_config(**kwargs)
        self.athena_compiled.update_config(**kwargs)
    
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