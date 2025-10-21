# pretrain_dataset_name = "philosopher"
# pretrain_dataset_hfpath = "datastax/philosopher-quotes"
# pretrain_dataset_hfdir = None
# pretrain_dataset_hfcolumn = "quote"
# pretrain_dataset_delimiter = "<|eos|>"
# pretrain_dataset_total_chars = 58524
# pretrain_dataset_valid_chars = pretrain_dataset_total_chars // 100

pretrain_dataset_name = "harrypotter"
pretrain_dataset_hfpath = "elricwan/HarryPotter"
pretrain_dataset_hfdir = "data"
pretrain_dataset_hfcolumn = "content"
pretrain_dataset_delimiter = "<|eos|>"
pretrain_dataset_total_chars = 12982412
pretrain_dataset_valid_chars = pretrain_dataset_total_chars // 100

# pretrain_dataset_name = "fineweb-edu"
# pretrain_dataset_hfpath = "HuggingFaceFW/fineweb-edu"
# pretrain_dataset_hfdir = "sample/10BT"
# pretrain_dataset_hfcolumn = "text"
# pretrain_dataset_delimiter = "<|eos|>"
# pretrain_dataset_total_chars = 45735998438 # found with the measure_dataset.py script
# pretrain_dataset_valid_chars = 100000

tokenizer_path = f"tokenizers/{pretrain_dataset_name}.json"

pretrain_dataset_train_chars = pretrain_dataset_total_chars - pretrain_dataset_valid_chars

assert(0 <= pretrain_dataset_total_chars)
assert(0 <= pretrain_dataset_valid_chars <= pretrain_dataset_total_chars)
assert(0 <= pretrain_dataset_train_chars <= pretrain_dataset_total_chars)
