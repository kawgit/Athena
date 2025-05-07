from argparse import ArgumentParser
import time
import torch

from athena.checkpoint import load_checkpoint
from athena.device import device
from athena.model import AthenaCompiled
from athena.tokenizer import tokenizer
from athena.utils import make_chat_pretty

argparser = ArgumentParser(description="Script for running inference on an athena model")
argparser.add_argument("--name", type=str, required=True)
args = argparser.parse_args()

athena = load_checkpoint(args.name)[0]
athena = AthenaCompiled(athena)

text = "Harry"
time_of_start = time.time()
time_of_last_print = time.time()
time_between_prints = 1
len_of_last_print = 0

text_tokens = tokenizer.encode(text)

with torch.inference_mode(), torch.autocast(device.type):
    for generation in athena.generate([text_tokens], 8 * athena.config.context_size, stream=True, temperature=0.3):

        time_elapsed = time.time() - time_of_last_print

        if time_elapsed > time_between_prints: 

            time_of_last_print = time.time()

            text = tokenizer.decode(generation.cum_tokens[0].tolist())
            curr_len = len(generation.cum_tokens[0])

            print("=" * 100)
            print(make_chat_pretty(text))
            print(f"Total tokens: {curr_len} Recent TPS: {(curr_len - len_of_last_print) / time_elapsed:.2f} Total TPS: {curr_len / (time.time() - time_of_start):.2f}")
            
            len_of_last_print = curr_len