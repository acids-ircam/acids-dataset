# 5. Partitions

`split` divides a dataset in partitions, that are datasets themselves :

```python
from acids_dataset.datasets import AudioDataset

dataset = AudioDataset("tutorial_dataset", sr=22050, chunk_length=1., meta_regexp=["**/{{instrument}}_*.wav"])
partitions = dataset.split({"train": 0.8, "valid": 0.2}, seed=0)
print({name: len(p) for name, p in partitions.items()})
```

Online datasets split files : chunks of a file always belong to the same partition. With `seed`, partitions are reproducible, which is essential in distributed training (every process must get the same partitions).

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
