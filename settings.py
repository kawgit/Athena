
# pretrain_dataset_name = "harrypotter"
# pretrain_dataset_hfpath = "elricwan/HarryPotter"
# pretrain_dataset_hfdir = "data"
# pretrain_dataset_hfcolumn = "content"

# pretrain_dataset_name = "openwebmath"
# pretrain_dataset_hfpath = "open-web-math/open-web-math"
# pretrain_dataset_hfsplit = "train"
# pretrain_dataset_hfcolumn = "text"

pretrain_dataset_name = "fineweb-edu"
pretrain_dataset_hfpath = "HuggingFaceFW/fineweb-edu"
pretrain_dataset_hfdir = "sample/10BT"
pretrain_dataset_hfcolumn = "text"
pretrain_dataset_delimiter = "<|eos|>"
pretrain_dataset_total_chars = 45735998438 # found with the measure_dataset.py script

tokenizer_path = f"tokenizers/{pretrain_dataset_name}.json"
