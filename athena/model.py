import math
import torch
import torch.nn as nn
import torch.nn.functional as functional

from athena.device import device
from athena.generation import AthenaGeneration
from athena.utils import save_tensor_as_image

class Athena(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.__dict__.update(config)
        
        self.max_tokens = round(self.context_size * self.context_multiple)
        
        self.register_buffer(
            "causal_buffer",
            torch.zeros((self.max_tokens, self.max_tokens), dtype=torch.bool),
            persistent=False
        )
        
        for q_index in range(self.max_tokens):
            self.causal_buffer[q_index][max(0, q_index + 1 - self.context_size):q_index + 1] = True

        self.register_buffer(
            "cos_buffer",
            torch.zeros((self.max_tokens, self.key_size), dtype=torch.float32),
            persistent=False
        )
        
        self.register_buffer(
            "sin_buffer",
            torch.zeros((self.max_tokens, self.key_size), dtype=torch.float32),
            persistent=False
        )
        
        i = torch.arange(self.key_size)
        half = self.key_size // 2
        offset = i % half

        exponent = (-2.0 / self.key_size) * offset
        scales = torch.pow(10000, exponent)

        m = torch.arange(self.max_tokens)
        angles = m[:, None] * scales[None, :]
        
        self.cos_buffer = torch.cos(angles)
        self.sin_buffer = torch.sin(angles)
        self.sin_buffer[:, :half] *= -1
        
        self.table = nn.Embedding(self.vocab_size, self.embedding_size)
        self.sas = nn.ModuleList([AthenaSA(self.config) for i in range(self.num_layers)])
        self.ffns = nn.ModuleList([AthenaFFN(self.config) for i in range(self.num_layers)])
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
        
        causal_buffer = self.causal_buffer[q_start:k_len, :k_len]
        cos_buffer = (self.cos_buffer[q_start:k_len], self.cos_buffer[:k_len])
        sin_buffer = (self.sin_buffer[q_start:k_len], self.sin_buffer[:k_len])
        
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
            # result["logits"] = 15 * result["logits"] / (result["logits"].square() + 225).sqrt()
        
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
        self.__dict__.update(config)

        self.norm = nn.RMSNorm(self.embedding_size, eps=1e-05, elementwise_affine=False)
        self.qkv_proj = nn.Linear(self.embedding_size, 2 * self.key_size * self.num_heads + self.head_size * self.num_heads, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_size, self.embedding_size, bias=False)

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.config[key] = value

    def forward(self, embeddings, cos_buffer, sin_buffer, causal_buffer, past_kv=None):

        normed = self.norm(embeddings) # B, C, E
        qkv = self.qkv_proj(normed) # B, C, H * Q + H * K + H * V

        keys_start = self.key_size * self.num_heads
        values_start = 2 * keys_start

        queries = qkv[..., :keys_start] # B, C, H * Q
        keys = qkv[..., keys_start:values_start] # B, C, H * K
        values = qkv[..., values_start:] # B, C, H * V
        
        batch_size, current_context_size, _ = embeddings.shape
        qk_shape = (batch_size, current_context_size, self.num_heads, self.key_size)
        v_shape = (batch_size, current_context_size, self.num_heads, self.head_size)

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
        
        attn = queries @ keys.transpose(-2, -1) / math.sqrt(self.key_size) # B, H, C, C
        attn = attn.masked_fill(causal_buffer == False, float("-inf"))
        attn = functional.softmax(attn, dim=-1)
        
        outputs = attn @ values # B, H, C, V
        
        outputs = outputs.transpose(1, 2).contiguous() # B, C, H, V
        outputs = outputs.flatten(-2) # B, C, H * V
        outputs = self.out_proj(outputs) # B, C, E

        return embeddings + outputs, present_kv

    def rotate(self, features, cos_buffer, sin_buffer):
        
        swapped = torch.cat((features[..., self.key_size // 2:], features[..., :self.key_size // 2]), dim=-1)
        
        return cos_buffer * features + sin_buffer * swapped

class AthenaFFN(nn.Module):
    def __init__(self, config):

        super().__init__()

        self.config = config
        self.__dict__.update(config)

        self.norm = nn.RMSNorm(self.embedding_size, eps=1e-05, elementwise_affine=False)
        self.up_proj = nn.Linear(self.embedding_size, 2 * self.hidden_size, bias=False)
        self.down_proj = nn.Linear(self.hidden_size, self.embedding_size, bias=False)
    
    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.config[key] = value
    
    def forward(self, embeddings):

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