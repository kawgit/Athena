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

from datasets import load_dataset
from torch.utils.data import DataLoader

def load_dataloader(tokenizer):
    tokenizer.padding_side = "left"
    dataset = load_dataset("open-r1/OpenR1-Math-220k", "default")["train"]
    dataset.set_format(type="torch", columns=["problem", "answer"])

    def collate_fn(batch):
        problems = tokenizer([entry["problem"] for entry in batch], padding=True, truncation=True, return_tensors="pt")
        answers = [entry["answer"] for entry in batch]

        return problems, answers

    return DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True, drop_last=True, collate_fn=collate_fn)