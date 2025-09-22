from datasets import load_dataset
from athena.dataloader import load_raw_dataset
from settings import pretrain_dataset_hfpath, pretrain_dataset_hfdir, pretrain_dataset_hfcolumn

ds = load_raw_dataset()

# Process in batches (avoids Python overhead)
def count_batch(batch):
    return {"counts": [len(x) for x in batch[pretrain_dataset_hfcolumn]]}

ds_with_counts = ds.map(count_batch, batched=True, batch_size=1000)
total = sum(ds_with_counts["counts"])
print(total)
