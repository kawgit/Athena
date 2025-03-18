from athena_tokenizer import load_tokenizer
from athena_dataloader import load_dataloader

tokenizer = load_tokenizer()
dataloader = load_dataloader(tokenizer)

for i, (a, b) in enumerate(dataloader):
    
    if i >= 10:
        break
    
    print(a['input_ids'].shape)

