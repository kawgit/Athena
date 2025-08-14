from argparse import ArgumentParser

from athena.checkpoint import load_checkpoint, save_checkpoint
from athena.graft import graft

argparser = ArgumentParser(description="Script for grafting the weights from one model to another of a different size. Weights in the destination model are partially overwritten and scaled down by a factor of 100.")
argparser.add_argument("--src_name", type=str, required=True)
argparser.add_argument("--dst_name", type=str, required=True)
argparser.add_argument("--dst_scale", type=float, default=.01)
argparser.add_argument("--layer_map", type=str, default="None")
args = argparser.parse_args()

src_athena, src_optimizer, src_scheduler = load_checkpoint(args.src_name)
dst_athena, dst_optimizer, dst_scheduler = load_checkpoint(args.dst_name)

graft(src_athena, dst_athena, dst_scale=args.dst_scale, layer_map=eval(args.layer_map))

save_checkpoint(dst_athena, scheduler=dst_scheduler)