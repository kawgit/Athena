from argparse import ArgumentParser
import os
import time
import torch

from athena.checkpoint import load_checkpoint

argparser = ArgumentParser(description="Script for printing the params in an Athena model")
argparser.add_argument("--name", type=str, required=True)
args = argparser.parse_args()

athena = load_checkpoint(args.name)[0]
for name, param in athena.named_parameters():
    print(name, param.shape)