from argparse import ArgumentParser
from athena.checkpoint import load_checkpoint
from athena.pretrain import Pretrainer

argparser = ArgumentParser(description="Script for pretraining an athena model from a given checkpoint")
argparser.add_argument("--name", type=str, required=True)
argparser.add_argument("--batch_size", type=int, default=4)
argparser.add_argument("--save_every", type=int, default=120)
argparser.add_argument("--valid_every", type=int, default=float("inf"))
argparser.add_argument("--epoch_limit", type=int, default=1)
argparser.add_argument("--time_limit", type=int, default=float("inf"))
args = argparser.parse_args()

athena, optimizer, scheduler = load_checkpoint(args.name)
pretrainer = Pretrainer(athena, optimizer, scheduler, batch_size=args.batch_size, save_every=args.save_every, valid_every=args.valid_every)
pretrainer.train(epoch_limit=args.epoch_limit, time_limit=args.time_limit)