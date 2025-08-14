import torch
from athena.tokenizer import tokenizer
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from datasets.config import HF_DATASETS_CACHE
from athena.utils import Timer
from settings import pretrain_dataset_name, pretrain_dataset_hfpath, pretrain_dataset_hfsplit, pretrain_dataset_hfcolumn
from torch.utils.data import DataLoader, Subset
import os
import random

def load_dataloader_pretrain(context_size, batch_size, resume_epoch=0):
    
    scrambled_dataset_name = f"{pretrain_dataset_name}_scrambled"
    cache_path = os.path.join(HF_DATASETS_CACHE, scrambled_dataset_name)
    
    if os.path.exists(cache_path):
        dataset = load_from_disk(cache_path)
    else:
        
        with Timer("Shuffling pretrain dataset"):
        
            dataset = load_dataset(pretrain_dataset_hfpath, split=pretrain_dataset_hfsplit)
            
            chunks = []
            for record in dataset:
                text = record[pretrain_dataset_hfcolumn]
                text_tokenized = tokenizer(text)["input_ids"]
                chunks.extend([text_tokenized[i : i + context_size] for i in range(0, len(text_tokenized) - context_size + 1, context_size)])
            random.shuffle(chunks)
            
            train_chunks = chunks[:int(len(chunks) * 0.98)]
            valid_chunks = chunks[int(len(chunks) * 0.98):]
            
            train_dataset = Dataset.from_dict({"input_ids": train_chunks})
            valid_dataset = Dataset.from_dict({"input_ids": valid_chunks})
            
            train_dataset.set_format(type="torch", columns=["input_ids"])
            valid_dataset.set_format(type="torch", columns=["input_ids"])
            
            dataset = DatasetDict({"train": train_dataset, "valid": valid_dataset})
            dataset.save_to_disk(cache_path)
    
    train_dataset = dataset["train"]
    valid_dataset = dataset["valid"]
    
    def collate_input_ids(batch):
        return torch.stack([example["input_ids"] for example in batch], dim=0)

    train_dataset = Subset(train_dataset, range(int(resume_epoch * len(train_dataset)) % len(train_dataset), len(train_dataset)))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_input_ids)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_input_ids)
    
    return train_dataloader, valid_dataloader

def load_dataloader_rl(split="train", batch_size=1):
    assert split in {"train", "valid"}, f"Invalid split: {split}"

    cache_path = os.path.join(HF_DATASETS_CACHE, "orca_math_split")
    split_path = os.path.join(cache_path, split)

    if os.path.exists(split_path):
        dataset = load_from_disk(split_path)
    else:
        full = load_dataset("microsoft/orca-math-word-problems-200k", "default")["train"]
        result = full.train_test_split(test_size=0.1, seed=42)
        split_map = {"train": result["train"], "valid": result["test"]}

        os.makedirs(cache_path, exist_ok=True)
        split_map["train"].save_to_disk(os.path.join(cache_path, "train"))
        split_map["valid"].save_to_disk(os.path.join(cache_path, "valid"))

        dataset = split_map[split]

    dataset.set_format(type="torch", columns=["question", "answer"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True)
