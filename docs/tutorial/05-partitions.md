# 5. Partitions

`split` divides a dataset in partitions, that are datasets themselves :

```python
from acids_dataset.datasets import AudioDataset

dataset = AudioDataset("tutorial_dataset", sr=22050, chunk_length=1., meta_regexp=["**/{{instrument}}_*.wav"])
partitions = dataset.split({"train": 0.8, "valid": 0.2}, seed=0)
print({name: len(p) for name, p in partitions.items()})
```

With `seed`, partitions are reproducible, which is essential in distributed training (every process must get the same partitions).

## Files or items

`by` chooses whether files or items are partitioned :

```python
by_files = dataset.split({"train": 0.8, "valid": 0.2}, by="files", seed=0)   # chunks of a file belong to the same partition
by_items = dataset.split({"train": 0.8, "valid": 0.2}, by="items", seed=0)   # chunks are partitioned independently
print({name: len(p) for name, p in by_files.items()}, {name: len(p) for name, p in by_items.items()})
```

Splitting by files avoids sharing audio between partitions, but with a few files partition sizes can be far from the ratios. With `by="auto"` (default), datasets are split by files, unless a partition would deviate from its ratio by more than `tolerance` (10 % by default) : they are then split by items.

`max_items` limits the size of some partitions, e.g. a validation set of at most 200 items, whatever the size of the dataset (remaining items go to the other partitions) :

```python
partitions = dataset.split({"train": 0.9, "valid": 0.1}, max_items={"valid": 200}, seed=0)
```

## Partitions by feature

With `features`, each value of a feature belongs to a single partition. With `balance_cardinality`, partition sizes follow the ratios as closely as possible :

```python
by_instrument = dataset.split({"train": 0.5, "valid": 0.5}, features=["instrument"], balance_cardinality=True, seed=0)
for name, p in by_instrument.items():
    print(name, sorted(p.feature_hash["instrument"].keys()))
```

Continuous features give *stratified* partitions : each partition gets a similar distribution of values.

```python
from acids_dataset.features import SpectralCentroid

dataset = AudioDataset("tutorial_dataset", sr=22050, chunk_length=1., features=[SpectralCentroid(sr=22050)])
stratified = dataset.split({"train": 0.7, "valid": 0.3}, features=["spectral_centroid"], n_bins=3, seed=0)
print({name: len(p.file_indices) for name, p in stratified.items()})
```

## Writing partitions

Partitions are written with `write`, and loaded back with `load_partition` :

```python
dataset = AudioDataset("tutorial_dataset", sr=22050, chunk_length=1.)
dataset.split({"train": 0.8, "valid": 0.2}, seed=0, write="split")
partitions = dataset.load_partition("split")
train_loader = partitions["train"].get_loader(batch_size=4)
print(len(train_loader))
```

Next : [pre-processing](06-preprocessing.md).

!!! info "See also"
    [Partitions](../guide/partitions.md) in the documentation.
