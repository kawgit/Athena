from argparse import ArgumentParser
from athena.config import AthenaConfig
from athena.device import device
from athena.model import Athena
from athena.pretrain import pretrain
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import ast

argparser = ArgumentParser(description="Script for initializing athena models with random weights")
argparser.add_argument("--packed_params", type=str, required=True)
args = argparser.parse_args()

params = ast.literal_eval(args.packed_params)

config = AthenaConfig()
config.vocab_size     = params[0]
config.context_size   = params[1]
config.embedding_size = params[2]
config.hidden_size    = params[3]
config.key_size       = params[4]
config.head_size      = params[5]
config.num_heads      = params[6]
config.num_layers     = params[7]
config.name = f"athena_{config.key_size}_{config.head_size}_{config.num_heads}_{config.num_layers}_{config.embedding_size}_{config.hidden_size}"
athena = Athena(config).to(device)

wandb.init(
    project="athena-pretrain",
    config=vars(argparser.parse_args()),
    reinit=True
)

optimizer = AdamW(athena.parameters(), lr=3e-5)
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

pretrain(athena, optimizer, scheduler, time_limit=6 * 60)
