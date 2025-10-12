from argparse import ArgumentParser
import os
import time
import torch

from athena.checkpoint import load_checkpoint
from athena.device import device
from athena.model import AthenaCompiled
from athena.tokenizer import tokenizer
from athena.utils import make_chat_pretty

argparser = ArgumentParser(description="Script for running inference on an athena model")
argparser.add_argument("--name", type=str, default=None)
args = argparser.parse_args()

athena = load_checkpoint(args.name)[0]
athena = AthenaCompiled(athena)

text = "In"
time_of_start = time.time()
time_of_last_print = time.time()
time_between_prints = 0.1
len_of_last_print = 0

text_tokens = tokenizer.encode(text)

with torch.inference_mode(), torch.autocast(device.type):
    try:
        for generation in athena.generate([text_tokens], 8 * athena.context_size, stream=True, temperature=.7):

                time_elapsed = time.time() - time_of_last_print

                if time_elapsed > time_between_prints:

                    time_of_last_print = time.time()

                    text = tokenizer.decode(generation.cum_tokens[0].tolist())
                    curr_len = len(generation.cum_tokens[0])
                    
                    recent_tps = (curr_len - len_of_last_print) / time_elapsed
                    total_tps = curr_len / (time.time() - time_of_start)

                    os.system("clear")
                    print(make_chat_pretty(text))
                    print(f"Total tokens: {curr_len} Recent TPS: {recent_tps:.2f} Total TPS: {total_tps:.2f}")
                    
                    len_of_last_print = curr_len
        
    except KeyboardInterrupt:
        os.system("clear && clear")
        print(make_chat_pretty(text))
        print(f"Total tokens: {curr_len} Recent TPS: {recent_tps:.2f} Total TPS: {total_tps:.2f}")
            
            