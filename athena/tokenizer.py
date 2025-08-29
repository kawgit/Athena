import torch
import os

from tokenizers import Tokenizer as RawTokenizer
from transformers import PreTrainedTokenizerFast

from settings import tokenizer_path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CustomTokenizer(PreTrainedTokenizerFast):

    def __init__(self, tokenizer_object, **kwargs):
        super().__init__(tokenizer_object=tokenizer_object, **kwargs)
        
        self.zero_token_id = self.convert_tokens_to_ids("0")
        self.tab_token_id = self.convert_tokens_to_ids("\t")
        self.space_token_id = self.convert_tokens_to_ids(" ")
        self.nospace_token_id = self.convert_tokens_to_ids("<|nospace|>")

    def __call__(self, text, padding=False, return_tensors=None, **kwargs):
        if isinstance(text, str):
            input_ids = self.encode(text, **kwargs)
            if return_tensors == "pt":
                return {"input_ids": torch.tensor([input_ids], dtype=torch.long)}
            return {"input_ids": input_ids}

        elif isinstance(text, list):
            encoded = [self.encode(t, **kwargs) for t in text]
            if padding:
                max_len = max(len(seq) for seq in encoded)
                encoded = [
                    [self.pad_token_id] * (max_len - len(seq)) + seq
                    for seq in encoded
                ]
            if return_tensors == "pt":
                return {"input_ids": torch.tensor(encoded, dtype=torch.long)}
            return {"input_ids": encoded}

        else:
            raise TypeError("Input must be a string or list of strings.")
    
    def isalnum(self, token_id):
        return self.zero_token_id <= token_id < self.tab_token_id
        
    def encode(self, text, **kwargs):
        segments = [seg for seg, bounds in self._tokenizer.pre_tokenizer.pre_tokenize_str(text)]
        token_ids = super().encode(segments, is_split_into_words=True, **kwargs)
        
        result = [token_ids[0]]
        
        last_isalnum = self.isalnum(token_ids[0])
        curr_isalnum = self.isalnum(token_ids[1]) if len(token_ids) >= 2 else None
        
        for i in range(1, len(token_ids)):
            next_isalnum = self.isalnum(token_ids[i+1]) if i + 1 < len(token_ids) else False
            
            if last_isalnum and curr_isalnum:
                result.append(self.nospace_token_id)

            if not (token_ids[i] == self.space_token_id and last_isalnum and next_isalnum):
                result.append(token_ids[i])

            last_isalnum = curr_isalnum
            curr_isalnum = next_isalnum
        
        return result

    def decode(self, token_ids, **kwargs):
        
        result = []
        
        for i, token_id in enumerate(token_ids):
            
            if i >= 1 and self.isalnum(token_id) and self.isalnum(token_ids[i-1]):
                result.append(self.space_token_id)
            
            if token_id != self.nospace_token_id:
                result.append(token_id)
            
        return "".join([self.convert_ids_to_tokens(i) for i in result])

tokenizer = RawTokenizer.from_file(tokenizer_path)
tokenizer = CustomTokenizer(tokenizer_object=tokenizer)
tokenizer.pad_token = "<|pad|>"
tokenizer.padding_side = "left"

unknown_token_id = tokenizer("<|unknown|>")['input_ids'][0]
pad_token_id     = tokenizer("<|pad|>")['input_ids'][0]
input_token_id   = tokenizer("<|input|>")['input_ids'][0]
output_token_id  = tokenizer("<|output|>")['input_ids'][0]
think_token_id   = tokenizer("<|think|>")['input_ids'][0]
say_token_id     = tokenizer("<|say|>")['input_ids'][0]
end_token_id     = input_token_id
