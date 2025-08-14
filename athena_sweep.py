from argparse import ArgumentParser
from athena.config import AthenaConfig
from athena.device import device
from athena.model import Athena
from athena.pretrain import pretrain
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb


argparser = ArgumentParser(description="Script for initializing athena models with random weights")
argparser.add_argument("--vocab_size",        type=int, default=32768)
argparser.add_argument("--context_size",      type=int, default=256)
argparser.add_argument("--embedding_size",    type=int, default=1536)
argparser.add_argument("--hidden_multiplier", type=int, default=4)
argparser.add_argument("--key_size",          type=int, default=64)
argparser.add_argument("--head_size",         type=int, default=96)
argparser.add_argument("--num_heads",         type=int, default=32)
argparser.add_argument("--num_layers",        type=int, default=3)
argparser.add_argument("--batch_size",        type=int, default=8)
argparser.add_argument("--lr",                type=float, default=3e-5)
args = argparser.parse_args()

config = AthenaConfig()
config.vocab_size     = args.vocab_size
config.context_size   = args.context_size
config.embedding_size = args.embedding_size
config.hidden_size    = args.embedding_size * args.hidden_multiplier
config.key_size       = args.key_size
config.head_size      = args.head_size
config.num_heads      = args.num_heads
config.num_layers     = args.num_layers
config.name = f"athena_{config.embedding_size}_{config.hidden_size}_{config.key_size}_{config.head_size}_{config.num_heads}_{config.num_layers}"
athena = Athena(config).to(device)

wandb.init(
    project="athena-pretrain",
    config=vars(athena.config),
    reinit=True
)

optimizer = AdamW(athena.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

pretrain(athena, optimizer, scheduler, batch_size=args.batch_size, epoch_limit=.2, time_limit=5*60)
