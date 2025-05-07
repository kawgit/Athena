from argparse import ArgumentParser
from athena.checkpoint import load_checkpoint
from athena.pretrain import pretrain

argparser = ArgumentParser(description="Script for pretraining an athena model from a given checkpoint")
argparser.add_argument("--name", type=str, required=True)
args = argparser.parse_args()

athena, optimizer, scheduler = load_checkpoint(args.name)
pretrain(athena, optimizer, scheduler)