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

import torch
from athena_tokenizer import load_tokenizer
from athena import load_athena
from device import device

torch.set_grad_enabled(False)

athena = torch.compile(load_athena().to(device)).eval()
tokenizer = load_tokenizer()

text = """<|system|>You are a helpful assistant.<|end|>"""

while True:

    text += f"<|user|>{input('<|user|> ')}<|end|><|assistant|>"
    text_tokens = tokenizer.encode(text)

    for text_tokens in athena.generate([text_tokens], 1000):

        text = tokenizer.decode(text_tokens[0])

        print("=" * 100, f" decoding {len(text_tokens[0])} tokens...")
        print(text.replace("<|end|>", "\n\n"))

    print("")

