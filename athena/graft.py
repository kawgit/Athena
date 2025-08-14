import math
import wandb

def graft(src_athena, dst_athena, dst_scale=.01, layer_map=None):
    
    assert(src_athena.embedding_size <= dst_athena.embedding_size)
    assert(src_athena.hidden_size <= dst_athena.hidden_size)
    assert(src_athena.num_layers <= dst_athena.num_layers)
    assert(src_athena.num_heads <= dst_athena.num_heads)
    assert(src_athena.head_size <= dst_athena.head_size)
    assert(src_athena.key_size == dst_athena.key_size)
    assert(src_athena.vocab_size == dst_athena.vocab_size)
    assert(src_athena.context_size == dst_athena.context_size)

    norm_scale = math.sqrt(src_athena.embedding_size / dst_athena.embedding_size)
    qk_scale = math.sqrt(math.sqrt(dst_athena.key_size / src_athena.key_size))

    dst_athena.table.weight.data *= dst_scale
    dst_athena.table.weight.data[:, :src_athena.embedding_size] = src_athena.table.weight.data
    dst_athena.vocab_proj.weight.data *= dst_scale
    dst_athena.vocab_proj.weight.data[:, :src_athena.embedding_size] = src_athena.vocab_proj.weight.data
    dst_athena.quality_proj.weight.data *= dst_scale
    dst_athena.quality_proj.weight.data[:, :src_athena.embedding_size] = src_athena.quality_proj.weight.data
    dst_athena.norm.weight.data[:src_athena.embedding_size] = src_athena.norm.weight.data * norm_scale

    layer_map = layer_map or {i : i for i in range(src_athena.num_layers)}
    layer_map_reversed = {v : k for k, v in layer_map.items()}

    for dst_i, (dst_sas, dst_ffn) in enumerate(zip(dst_athena.sas, dst_athena.ffns)):
        
        dst_sas.out_proj.weight.data *= dst_scale
        dst_ffn.down_proj.weight.data *= dst_scale
        
        if dst_i not in layer_map_reversed:
            continue
        
        src_i = layer_map_reversed[dst_i]
        src_sas = src_athena.sas[src_i]
        src_ffn = src_athena.ffns[src_i]
        
        dst_ffn.norm.weight.data[:src_ffn.embedding_size] = src_ffn.norm.weight.data * norm_scale
        dst_sas.norm.weight.data[:src_sas.embedding_size] = src_sas.norm.weight.data * norm_scale
        
        dst_ffn.up_proj.weight.data[:src_ffn.hidden_size, :src_ffn.embedding_size] = src_ffn.up_proj.weight.data[:src_ffn.hidden_size]
        dst_ffn.up_proj.weight.data[dst_ffn.hidden_size : dst_ffn.hidden_size + src_ffn.hidden_size, :src_ffn.embedding_size] = src_ffn.up_proj.weight.data[src_ffn.hidden_size:]
        dst_ffn.down_proj.weight.data[:src_ffn.embedding_size, :src_ffn.hidden_size] = src_ffn.down_proj.weight.data
        

        for head_index in range(src_sas.num_heads):
            
            src_queries_start = head_index * src_sas.key_size
            dst_queries_start = head_index * dst_sas.key_size
            dst_sas.qkv_proj.weight.data[dst_queries_start : dst_queries_start + dst_sas.key_size] *= dst_scale
            dst_sas.qkv_proj.weight.data[dst_queries_start : dst_queries_start + src_sas.key_size, :src_sas.embedding_size] = src_sas.qkv_proj.weight.data[src_queries_start : src_queries_start + src_sas.key_size] * qk_scale

            src_keys_start = head_index * src_sas.key_size + src_sas.num_heads * src_sas.key_size
            dst_keys_start = head_index * dst_sas.key_size + dst_sas.num_heads * dst_sas.key_size
            dst_sas.qkv_proj.weight.data[dst_keys_start : dst_keys_start + dst_sas.key_size] *= dst_scale
            dst_sas.qkv_proj.weight.data[dst_keys_start : dst_keys_start + src_sas.key_size, :src_sas.embedding_size] = src_sas.qkv_proj.weight.data[src_keys_start : src_keys_start + src_sas.key_size] * qk_scale

            src_values_start = head_index * src_sas.head_size + 2 * src_sas.num_heads * src_sas.key_size
            dst_values_start = head_index * dst_sas.head_size + 2 * dst_sas.num_heads * dst_sas.key_size
            dst_sas.qkv_proj.weight.data[dst_values_start : dst_values_start + dst_sas.head_size] *= dst_scale
            dst_sas.qkv_proj.weight.data[dst_values_start : dst_values_start + src_sas.head_size, :src_sas.embedding_size] = src_sas.qkv_proj.weight.data[src_values_start : src_values_start + src_sas.head_size]
            
            src_outputs_start = head_index * src_sas.head_size
            dst_outputs_start = head_index * dst_sas.head_size
            dst_sas.out_proj.weight.data[:src_sas.embedding_size, dst_outputs_start : dst_outputs_start + src_sas.head_size] = src_sas.out_proj.weight.data[:, src_outputs_start : src_outputs_start + src_sas.head_size]
    
    api = wandb.Api()
    src_run = api.run(f"kawgit56-none/athena-pretrain/{src_athena.wandb_id}")
    
    dst_run = wandb.init(
        project="athena-pretrain",
        name=dst_athena.name,
        config=dst_athena.config
    )
    dst_run.log(dict(src_run.summary))
    dst_run.finish()
    
    dst_athena.update_config(wandb_id=dst_run.id)
    