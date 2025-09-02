from datasets import load_dataset
from settings import pretrain_dataset_hfpath, pretrain_dataset_hfdir, pretrain_dataset_hfcolumn

ds = load_dataset(
    pretrain_dataset_hfpath,
    data_dir=pretrain_dataset_hfdir,
    split="train",
)

# Process in batches (avoids Python overhead)
def count_batch(batch):
    return {"counts": [len(x) for x in batch[pretrain_dataset_hfcolumn]]}

ds_with_counts = ds.map(count_batch, batched=True, batch_size=1000)
total = ds_with_counts["counts"].sum()
print(total)
