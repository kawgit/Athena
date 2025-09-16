import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from typing import Optional, List

from settings import (
    pretrain_dataset_hfpath,
    pretrain_dataset_hfdir,
    pretrain_dataset_hfcolumn,
    pretrain_dataset_delimiter,
)
from athena.tokenizer import tokenizer


def _encode_ids(text: str) -> List[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)

def _delimiter_str() -> str:
    return pretrain_dataset_delimiter or ""


class CharOffsetChunkIterable(IterableDataset):
    """
    Character-sliced global stream -> tokenized -> emits fixed-length token chunks of size (C+1).
    Each yielded item is a 1D LongTensor of length (context_size+1).
    """
    def __init__(
        self,
        context_size: int,
        start_offset_chars: int = 0,
        take_chars: Optional[int] = None,
    ):
        super().__init__()
        self.context_size = int(context_size)
        self.emit_len = self.context_size + 1
        self.start_offset_chars = max(0, int(start_offset_chars))
        self.curr_offset_chars = self.start_offset_chars
        self.take_chars = None if take_chars is None else int(take_chars)
        self.delim = _delimiter_str()

    def __iter__(self):
        ds = load_dataset(
            pretrain_dataset_hfpath,
            data_dir=pretrain_dataset_hfdir,
            split="train",
        )

        skip = self.start_offset_chars
        remaining_chars = self.take_chars  # None => unlimited
        token_buf: List[int] = []

        for ex in ds:
            segment = (ex[pretrain_dataset_hfcolumn] or "") + self.delim
            seg_len = len(segment)

            # Fast character skip
            if skip:
                if skip >= seg_len:
                    skip -= seg_len
                    continue
                segment = segment[skip:]
                seg_len = len(segment)
                skip = 0

            # Character take budget
            if remaining_chars is not None:
                if remaining_chars <= 0:
                    break
                if remaining_chars < seg_len:
                    segment = segment[:remaining_chars]
                    seg_len = len(segment)
                    remaining_chars = 0
                else:
                    remaining_chars -= seg_len

            # Tokenize only the kept slice
            ids = _encode_ids(segment)
            token_buf.extend(ids)
            
            # Linearly interpolate between last char count and next char count
            num_tokens = len(token_buf)
            num_emits = (num_tokens - 1) // self.context_size
            
            if num_emits == 0:
                self.curr_offset_chars += seg_len
                continue
            
            slope = seg_len / num_emits
            original_offset = self.curr_offset_chars

            # Emit fixed-length (C+1) chunks; stride = C (overlap by 1)
            while len(token_buf) >= self.emit_len:
                chunk = token_buf[:self.emit_len]
                yield torch.tensor(chunk, dtype=torch.long)  # shape: (C+1,)
                token_buf = token_buf[self.context_size:]    # keep last token for next shift
                
                # Update current offset
                self.curr_offset_chars += round(slope)
                self.curr_offset_chars = min(self.curr_offset_chars, original_offset + seg_len)
    
def _collate_tokens(batch: List[torch.Tensor]) -> torch.Tensor:
    # Stacks to shape (B, C+1)
    return torch.stack(batch, dim=0)


def load_dataloader_pretrain(
    context_size: int,
    batch_size: int,
    valid_chars: int,      # first N chars go to validation
    resume_chars: int = 0  # then skip this many chars before training
):
    """
    Whole dataset is treated as one big string with a delimiter between records.
    Validation takes the first `valid_chars` characters.
    Training starts at offset `valid_chars + resume_chars` and streams the rest.

    Returns:
        train_loader, valid_loader
        where each batch is a LongTensor of shape (B, C+1)
    """
    if valid_chars < 0 or resume_chars < 0:
        raise ValueError("valid_chars and resume_chars must be >= 0")

    valid_iterable = CharOffsetChunkIterable(
        context_size=context_size,
        start_offset_chars=0,
        take_chars=valid_chars,
    )
    train_iterable = CharOffsetChunkIterable(
        context_size=context_size,
        start_offset_chars=valid_chars + resume_chars,
        take_chars=None,
    )

    train_loader = DataLoader(
        train_iterable,
        batch_size=batch_size,
        collate_fn=_collate_tokens,
        num_workers=0,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_iterable,
        batch_size=batch_size,
        collate_fn=_collate_tokens,
        num_workers=0,
        drop_last=True,
    )
    return train_loader, valid_loader
