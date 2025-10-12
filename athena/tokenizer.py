from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from settings import tokenizer_path

# load raw tokenizer
tok = Tokenizer.from_file(tokenizer_path)

# wrap for transformers
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tok,
    unk_token="<|unknown|>",
    eos_token="<|eos|>"
)
