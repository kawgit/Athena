import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from typing import Optional, List, Tuple

from settings import (
    pretrain_dataset_hfpath,
    pretrain_dataset_hfdir,
    pretrain_dataset_hfcolumn,
    pretrain_dataset_delimiter,
    pretrain_dataset_total_chars,
    pretrain_dataset_train_chars,
    pretrain_dataset_valid_chars,
)

def load_raw_dataset():
    return load_dataset(
        pretrain_dataset_hfpath,
        data_dir=pretrain_dataset_hfdir,
        split="train",
    )

def _encode_ids(text: str) -> List[int]:
    from athena.tokenizer import tokenizer
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)

def _delimiter_str() -> str:
    return pretrain_dataset_delimiter or ""


class CharOffsetChunkIterable(IterableDataset):
    """
    Character-sliced global stream -> tokenized -> emits fixed-length token chunks of size (C+1).
    Each yielded item is a tuple: (curr_offset_chars: int, 1D LongTensor of length (context_size+1)).
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
        ds = load_raw_dataset()

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

            # Plan emits for this segment
            num_tokens = len(token_buf)
            num_emits = (num_tokens - 1) // self.context_size

            if num_emits == 0:
                self.curr_offset_chars += seg_len
                continue

            # Evenly interpolate offsets within this segment
            original_offset = self.curr_offset_chars
            emits_done_in_seg = 0
            slope = seg_len / max(1, num_emits)

            # Emit fixed-length (C+1) chunks; stride = C (overlap by 1)
            while len(token_buf) >= self.emit_len:
                # Emit current offset alongside the tokens (offset BEFORE updating)
                offset_to_emit = self.curr_offset_chars
                chunk = token_buf[:self.emit_len]
                yield (offset_to_emit, torch.tensor(chunk, dtype=torch.long))  # (int, (C+1,))

                token_buf = token_buf[self.context_size:]  # keep last token for next shift

                # Update current offset with evenly spaced targets in this segment
                emits_done_in_seg += 1
                target = original_offset + round(emits_done_in_seg * slope)
                self.curr_offset_chars = min(target, original_offset + seg_len)

def _collate_tokens(batch: List[Tuple[int, torch.Tensor]]) -> Tuple[int, torch.Tensor]:
    """
    Collate a list of (offset:int, tokens:Tensor[C+1]) into:
      (max_offset:int, tokens:Tensor[B, C+1])
    """
    # batch: List[(offset, tensor)]
    offsets, tensors = zip(*batch)
    max_offset = int(max(offsets)) if offsets else 0
    return max_offset, torch.stack(list(tensors), dim=0)


def load_dataloader_pretrain(
    context_size: int,
    train_batch_size: int,
    valid_batch_size: int,
    resume_chars: int = 0  # then skip this many chars before training
):
    """
    Whole dataset is treated as one big string with a delimiter between records.
    Validation takes the first `pretrain_dataset_valid_chars` characters.
    Training starts at offset `pretrain_dataset_valid_chars + resume_chars` and streams
    at most `pretrain_dataset_train_chars - resume_chars` characters.

    Returns:
        train_loader, valid_loader
        where each batch is a tuple:
          (max_offset_in_batch: int, LongTensor of shape (B, C+1))
    """
    if pretrain_dataset_valid_chars < 0 or resume_chars < 0:
        raise ValueError("pretrain_dataset_valid_chars and resume_chars must be >= 0")

    # Optional sanity checks if total known
    if pretrain_dataset_total_chars is not None:
        if pretrain_dataset_valid_chars + pretrain_dataset_train_chars > pretrain_dataset_total_chars:
            raise ValueError("valid_chars + train_chars exceeds pretrain_dataset_total_chars")
        if resume_chars > pretrain_dataset_train_chars:
            raise ValueError("resume_chars cannot exceed pretrain_dataset_train_chars")

    # Bound the training window after resume
    train_take = max(0, pretrain_dataset_train_chars - resume_chars)

    train_iterable = CharOffsetChunkIterable(
        context_size=context_size,
        start_offset_chars=pretrain_dataset_valid_chars + resume_chars,
        take_chars=train_take,  # bounded
    )

    valid_iterable = CharOffsetChunkIterable(
        context_size=context_size,
        start_offset_chars=0,
        take_chars=pretrain_dataset_valid_chars,
    )

    train_loader = DataLoader(
        train_iterable,
        batch_size=train_batch_size,
        collate_fn=_collate_tokens,
        num_workers=0,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_iterable,
        batch_size=valid_batch_size,
        collate_fn=_collate_tokens,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, valid_loader
