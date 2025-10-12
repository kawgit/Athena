#!/usr/bin/env python3
# train_tokenizer.py
import math
import argparse
import random
from typing import Iterator, List, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from athena.dataloader import load_raw_dataset
from settings import tokenizer_path, pretrain_dataset_hfcolumn


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer with byte-fallback using Athena dataset.")
    p.add_argument("--vocab_size", type=int, default=32768, help="Target vocabulary size.")
    p.add_argument("--min_frequency", type=int, default=2, help="Min freq for merges.")
    p.add_argument("--dataset_fraction", type=float, default=1.0, help="Fraction of dataset to use in (0,1]. E.g., 0.1 = 10%%.")
    p.add_argument("--shuffle", action="store_true", help="Shuffle dataset before sampling fraction.")
    p.add_argument("--seed", type=int, default=1337, help="Shuffle seed (if --shuffle).")
    p.add_argument("--add_prefix_space", type=bool, default=True, help="Use GPT-2/Roberta-style leading-space handling (recommended).")
    return p


def select_indices(n_total: int, fraction: float, shuffle: bool, seed: int) -> List[int]:
    if not (0 < fraction <= 1):
        raise ValueError("--dataset_fraction must be in (0, 1]. Got: %r" % fraction)
    n_sample = max(1, math.ceil(fraction * n_total))
    idxs = list(range(n_total))
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(idxs)
    return idxs[:n_sample]


def get_training_corpus(dataset, col: str, indices: List[int]) -> Iterator[str]:
    # Yield raw text exactly as stored (no normalization/strip) to preserve code/whitespace
    for i in indices:
        ex = dataset[i]
        text = ex.get(col, None)
        if text is None:
            continue
        # Ensure it's a string (HF datasets sometimes store non-str types)
        yield str(text)


def main():
    args = build_argparser().parse_args()

    # 1) Load dataset and pick subset
    dataset = load_raw_dataset()
    n_total = len(dataset)
    indices = select_indices(n_total, args.dataset_fraction, args.shuffle, args.seed)
    n_sample = len(indices)

    # 2) Define tokenizer (byte-level BPE with byte fallback; only <|eos|>, <|unknown|>)
    tokenizer = Tokenizer(BPE(unk_token="<|unknown|>", byte_fallback=True))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=args.add_prefix_space)

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["<|eos|>", "<|unknown|>"],
        show_progress=True,
    )

    # 3) Train from iterator (stream from chosen subset)
    corpus_iter = get_training_corpus(dataset, pretrain_dataset_hfcolumn, indices)
    tokenizer.train_from_iterator(corpus_iter, trainer=trainer, length=n_sample)

    # 4) Post-processor + decoder for lossless round-trip
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    tokenizer.decoder = ByteLevelDecoder()

    # 5) Save
    tokenizer.save(tokenizer_path)

    # 6) Report
    used_pct = 100.0 * n_sample / n_total if n_total > 0 else 0.0
    print(f"✅ Trained tokenizer on {n_sample}/{n_total} samples ({used_pct:.1f}%).")
    print(f"   vocab_size={args.vocab_size}, min_frequency={args.min_frequency}, "
          f"add_prefix_space={args.add_prefix_space}, shuffle={args.shuffle}, seed={args.seed}")
    print(f"💾 Saved to: {tokenizer_path}")


if __name__ == "__main__":
    main()
