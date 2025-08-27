from argparse import ArgumentParser
from athena.checkpoint import save_checkpoint
from athena.model import Athena
from athena.utils import generate_model_name

argparser = ArgumentParser(description="Script for initializing athena models with random weights")
argparser.add_argument("--name", type=str, default=None)
argparser.add_argument("--embedding_size", type=int, default=1024)
argparser.add_argument("--hidden_size", type=int, default=2048)
argparser.add_argument("--num_layers", type=int, default=4)
argparser.add_argument("--num_heads", type=int, default=8)
argparser.add_argument("--head_size", type=int, default=16)
argparser.add_argument("--key_size", type=int, default=8)
argparser.add_argument("--vocab_size", type=int, default=32768)
argparser.add_argument("--context_size", type=int, default=256)
argparser.add_argument("--context_multiple", type=float, default=2)
argparser.add_argument("--null_len", type=int, default=4)

config = vars(argparser.parse_args())
config["wandb_id"] = None
config["name"] = config["name"] or generate_model_name(config)

athena = Athena(config)
save_checkpoint(athena)

