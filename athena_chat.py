import time
import torch

from athena.checkpoint import load_checkpoint
from athena.device import device
from athena.model import AthenaCompiled
from athena.tokenizer import tokenizer
from athena.utils import make_chat_pretty
from settings import checkpoint_path

torch.set_grad_enabled(False)

athena, optimizer = load_checkpoint(checkpoint_path)
athena = AthenaCompiled(athena)
athena.eval()

text = ""
time_of_last_print = 0
time_between_prints = 1 / 3
len_of_last_print = 0

while True:

    user_input = input("Input:\n\t")
    text += f"<|input|>{user_input}<|output|>"
    text_tokens = tokenizer.encode(text)

    for generation in athena.generate([text_tokens], 1000, stream=True):

        time_elapsed = time.time() - time_of_last_print

        if time_elapsed > time_between_prints:

            time_of_last_print = time.time()
            

            text = tokenizer.decode(generation.cum_tokens[0])
            curr_len = len(generation.cum_tokens[0])

            print("=" * 100)
            print(make_chat_pretty(text))
            print(f"Total tokens: {curr_len} Recent TPS: {(curr_len - len_of_last_print) / time_elapsed}")
            
            len_of_last_print = curr_len

