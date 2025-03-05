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

seed = """<|system|>You are a helpful assistant.<|end|>
<|user|>Solve -42*r + 27*c = -1167 and 130*r + 4*c = 372 for r.<|end|>
<|assistant|>
"""

athena = torch.compile(load_athena().to(device))
tokenizer = load_tokenizer()
text_tokens = tokenizer.encode(seed)

end_token_ids = list(tokenizer.encode("<|end|><|endoftext|>"))

for new_token in athena.generate(text_tokens, 300, end_token_ids=end_token_ids):
    text_tokens.append(new_token)

    print("=" * 100)
    print(tokenizer.decode(text_tokens))

print("")