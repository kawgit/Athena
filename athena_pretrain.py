from argparse import ArgumentParser
from athena.checkpoint import load_checkpoint
from athena.pretrain import Pretrainer
from athena.kuon import attach_kuon_hooks
from athena.utils import seed_all

seed_all(0)

argparser = ArgumentParser(description="Script for pretraining an athena model from a given checkpoint")
argparser.add_argument("--name", type=str, default=None)
argparser.add_argument("--train_batch_size", type=int, default=1)
argparser.add_argument("--valid_batch_size", type=int, default=2)
argparser.add_argument("--backwards_every", type=int, default=1)
argparser.add_argument("--log_every", type=int, default=0)
argparser.add_argument("--save_every", type=int, default=120)
argparser.add_argument("--valid_every", type=int, default=float("inf"))
argparser.add_argument("--epoch_limit", type=float, default=float("inf"))
argparser.add_argument("--time_limit", type=int, default=float("inf"))
args = argparser.parse_args()

athena, optimizer, scaler, autocast_ctx = load_checkpoint(args.name)
attach_kuon_hooks(athena)
pretrainer = Pretrainer(athena, optimizer, scaler, autocast_ctx, train_batch_size=args.train_batch_size, valid_batch_size=args.valid_batch_size, backwards_every=args.backwards_every, log_every=args.log_every, save_every=args.save_every, valid_every=args.valid_every)
pretrainer.train(epoch_limit=args.epoch_limit, time_limit=args.time_limit)