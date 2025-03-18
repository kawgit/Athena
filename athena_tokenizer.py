from transformers import AutoTokenizer
from settings import hfmodel_name

def load_tokenizer():
    return AutoTokenizer.from_pretrained(hfmodel_name)

padding_token_id = 32000
end_token_id = 32007
