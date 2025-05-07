from argparse import ArgumentParser
from athena.checkpoint import save_checkpoint
from athena.config import AthenaConfig
from athena.model import Athena
import wandb

argparser = ArgumentParser(description="Script for initializing athena models with random weights")
argparser.add_argument("--name",           type=str, default=None)
argparser.add_argument("--vocab_size",     type=int, default=32768)
argparser.add_argument("--context_size",   type=int, default=256)
argparser.add_argument("--embedding_size", type=int, default=1536)
argparser.add_argument("--hidden_size",    type=int, default=4096)
argparser.add_argument("--key_size",       type=int, default=64)
argparser.add_argument("--head_size",      type=int, default=96)
argparser.add_argument("--num_heads",      type=int, default=16)
argparser.add_argument("--num_layers",     type=int, default=3)
args = argparser.parse_args()

config = AthenaConfig()
config.name           = args.name
config.vocab_size     = args.vocab_size
config.context_size   = args.context_size
config.embedding_size = args.embedding_size
config.hidden_size    = args.hidden_size
config.key_size       = args.key_size
config.head_size      = args.head_size
config.num_heads      = args.num_heads
config.num_layers     = args.num_layers
config.wandb_id       = wandb.util.generate_id()
athena = Athena(config)

if config.name == None:
    config.name = f"athena_{config.key_size}_{config.head_size}_{config.num_heads}_{config.num_layers}_{config.embedding_size}_{config.hidden_size}"

save_checkpoint(athena)

