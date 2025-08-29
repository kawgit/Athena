from string import ascii_letters, digits, punctuation
from tokenizers import Regex
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer
import json
import os

from athena.dataloader import load_raw_dataset
from settings import tokenizer_path, pretrain_dataset_hfcolumn

dataset = load_raw_dataset()
def get_training_corpus():
    for example in dataset:
        yield example[pretrain_dataset_hfcolumn]

tokenizer = Tokenizer(BPE(unk_token="<|unknown|>", byte_fallback=True))
tokenizer.normalizer = NFKC()
tokenizer.pre_tokenizer = Split(Regex("(\\w+(['’‘ʼʾʿ＇ꞌ՚᾿´`]\\w+)*)|((\\w+['’‘ʼʾʿ＇ꞌ՚᾿´`])*\\w+)"), behavior="isolated")

trainer = BpeTrainer(
    vocab_size = 32768,
    min_frequency = 2,
    initial_alphabet = list(
        ascii_letters
        + digits
        + punctuation
        + "+-=*/^()[]{}<>|%&≈≠≤≥±√π∞"
        + " \n\t"
    ),
    special_tokens = [
        "<|pad|>",
        "<|unknown|>",
        "<|input|>",
        "<|output|>",
        "<|think|>",
        "<|say|>",
        "<|nospace|>",
    ],
)

tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

if not os.path.exists(os.path.dirname(tokenizer_path)):
    os.mkdir(os.path.dirname(tokenizer_path))

tokenizer.save(tokenizer_path)

tokenizer = Tokenizer.from_file(tokenizer_path)
with open(tokenizer_path, "r", encoding="utf-8") as f:
    data = json.load(f)

vocab = data["model"]["vocab"]
merges = data["model"]["merges"]

special_tokens = [tok for tok in vocab.keys() if tok.startswith("<|") and tok.endswith("|>")]
regular_tokens = [tok for tok in vocab.keys() if tok not in special_tokens]

word_tokens  = []
other_tokens = []

for tok in regular_tokens:

    if any(c.isalnum() for c in tok):
        word_tokens.append(tok)
    else:
        other_tokens.append(tok)

sorted_vocab = special_tokens + word_tokens + other_tokens

new_vocab = {tok: i for i, tok in enumerate(sorted_vocab)}
new_merges = [tuple(m) for m in merges]
tokenizer.model = BPE(new_vocab, new_merges)
tokenizer.save(tokenizer_path)
