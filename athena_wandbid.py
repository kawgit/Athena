from athena.checkpoint import load_checkpoint, save_checkpoint
from argparse import ArgumentParser

argparser = ArgumentParser(description="Script for adding a wandb to an incorrectly initialized checkpoint")
argparser.add_argument("--name", type=str, required=True)
argparser.add_argument("--wandb_id", type=str, required=True)
args = argparser.parse_args()

athena, optimizer, scheduler = load_checkpoint(args.name)
athena.config.wandb_id = args.wandb_id
save_checkpoint(athena, optimizer, scheduler)