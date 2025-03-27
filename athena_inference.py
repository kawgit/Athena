#    Copyright 2025 Kenneth Wilber (kawgit)

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import time
import torch

from athena_tokenizer import load_tokenizer
from athena import load_athena
from device import device

torch.set_grad_enabled(False)

athena = torch.compile(load_athena().to(device)).eval()
tokenizer = load_tokenizer()

batch_size = 3
seed = """<|system|>You are a helpful assistant.<|end|>
<|user|>Solve -42*r + 27*c = -1167 and 130*r + 4*c = 372.<|end|>
<|assistant|>
"""
reply = """<|user|>Your answer may or may not be correct. Please double check your calculations or try again. <|end|><|assistant|>"""

text_tokenss = [tokenizer.encode(seed) for i in range(batch_size)]
reply_queue = [[tokenizer.encode(reply) for i in range(1)] for i in range(batch_size)]
start_time = time.time()
time_of_last_print = 0
time_between_prints = .5

for text_tokenss in athena.generate(text_tokenss, 4000, reply_queue=reply_queue):

    if time.time() - time_of_last_print > time_between_prints:

        time_of_last_print = time.time()

        print("*=" * 50)

        for text_tokens in text_tokenss:
            print("=" * 100, f" decoding {len(text_tokens)} tokens...")
            print(tokenizer.decode(text_tokens))

print("")

num_tokens = sum([len(text_tokens) for text_tokens in text_tokenss])

print("Tokens:", num_tokens)
print("Seconds elapsed:", time.time() - start_time)
print("Tokens per second:", num_tokens / (time.time() - start_time))