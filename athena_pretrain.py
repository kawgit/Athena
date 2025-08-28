from argparse import ArgumentParser
from athena.checkpoint import load_checkpoint, save_checkpoint
from athena.pretrain import Pretrainer

argparser = ArgumentParser(description="Script for pretraining an athena model from a given checkpoint")
argparser.add_argument("--name", type=str, required=True)
argparser.add_argument("--batch_size", type=int, default=2)
argparser.add_argument("--log_every", type=int, default=0)
argparser.add_argument("--save_every", type=int, default=120)
argparser.add_argument("--valid_every", type=int, default=float("inf"))
argparser.add_argument("--epoch_limit", type=int, default=1)
argparser.add_argument("--time_limit", type=int, default=float("inf"))
args = argparser.parse_args()

athena, optimizer, scheduler, scaler, autocast_ctx = load_checkpoint(args.name)
pretrainer = Pretrainer(athena, optimizer, scheduler, scaler, autocast_ctx, batch_size=args.batch_size, log_every=args.log_every, save_every=args.save_every, valid_every=args.valid_every)
pretrainer.train(epoch_limit=args.epoch_limit, time_limit=args.time_limit)