from athena.checkpoint import load_checkpoint, save_checkpoint

source_name = "planck"
dest_name = "planck_tiny"

def config_modifier(config):
    config.num_layers = 4
    config.wandb_id = None

athena, optimizer = load_checkpoint(f"checkpoints/{source_name}.ckpt", config_modifier=config_modifier)

for ffn in athena.ffns[-4:]:
    ffn.down_proj.weight.data /= 1000

for sa in athena.sas[-4:]:
    sa.out_proj.weight.data /= 1000

save_checkpoint(f"checkpoints/{dest_name}.ckpt", athena, optimizer)