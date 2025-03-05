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
from tokenizer import load_tokenizer
from athena import load_athena
from device import device

athena = torch.compile(load_athena().to(device))
tokenizer = load_tokenizer()
end_token_ids = list(tokenizer.encode("<|end|><|endoftext|>"))

text = """<|system|>You are a helpful assistant.<|end|>"""

while True:

    text += f"<|user|>{input('<|user|> ')}<|end|><|assistant|>"
    text_tokens = tokenizer.encode(text)

    for new_token in athena.generate(text_tokens, 300, end_token_ids=end_token_ids):
        text_tokens.append(new_token)
        text = tokenizer.decode(text_tokens)

        print("=" * 100)
        print(text.replace("<|end|>", "\n\n"))

    print("")

