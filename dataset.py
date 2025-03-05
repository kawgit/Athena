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

from torch.utils.data import Dataset
import numpy as np
import os
import torch

from settings import context_size, dataset_name
from tokenizer import load_tokenizer
from utils import load_file

class TransformerDataset(Dataset):

    def __init__(self, text_indexes):

        self.text_indexes = text_indexes

    def __len__(self):

        return len(self.text_indexes) - context_size

    def __getitem__(self, idx):
        return self.text_indexes[idx:idx+context_size], self.text_indexes[idx+1:idx+context_size+1]
    

def load_dataset():

    if os.path.exists(f"datasets/{dataset_name}.npy"):
        text_indexes = np.load(f"datasets/{dataset_name}.npy")
    else:
        text = load_file(f"datasets/{dataset_name}.txt")
        tokenizer = load_tokenizer()
        text_indexes = tokenizer.encode(text, use_pbar=True)
        np.save(f"datasets/{dataset_name}.npy", text_indexes)

    return TransformerDataset(torch.tensor(text_indexes))