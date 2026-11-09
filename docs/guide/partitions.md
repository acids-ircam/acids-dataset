# Partitions

Datasets can be split into partitions, randomly or according to features (such that each feature value belongs to a single partition) :

```python
partitions = dataset.split({"train": 0.8, "valid": 0.2})                                   # random split
partitions = dataset.split({"train": 0.8, "valid": 0.2}, features=["original_path"])       # chunks of a file stay in the same partition
partitions = dataset.split({"train": 0.8, "valid": 0.2}, features=["instrument"], 
                           balance_cardinality=True, tolerance=0.1)                         # balance partition sizes
partitions = dataset.split({"train": 0.8, "valid": 0.2}, features=["loudness"], n_bins=10) # stratified over a continuous feature
partitions = dataset.split({"train": 0.8, "valid": 0.2}, seed=0)                          # reproducible split
partitions = dataset.split({"train": 0.8, "valid": 0.2}, write="my_split")                 # write partition
partitions = dataset.load_partition("my_split")                                             # load it back
train_loader = partitions["train"].get_loader(batch_size=16)
```

- Partitions are drawn with the global random state, unless `seed` is given : partitions with the same seed are identical, including the order of their items. **For distributed training, give the same seed in every process** (otherwise each process gets different partitions, mixing training and validation data), or split and write partitions in one process and load them in the others.
- Continuous features (e.g. `loudness`, or any feature that is not hashed) give stratified partitions : items (or files in online mode, or groups of categorical features, e.g. `features=["original_path", "loudness"]`) are sorted in `n_bins` quantile bins of the mean value of the feature (histogram), and distributed among partitions such that each partition has a similar distribution of the feature. At most one continuous feature can be given. In online mode, feature values are computed on the items of each file, which may take a while.
- With `balance_cardinality=True`, feature values are distributed such that partition sizes follow the given ratios (with a warning if a partition exceeds `tolerance`). If ratios sum to 1, every item belongs to a partition ; otherwise, the remaining part is left out.
- If a `partition` feature exists (e.g. `--meta_regexp "**/{{partition}}/*.wav"` for `train/` and `valid/` subfolders), its values are used as partitions (disable with `use_meta_if_available=False`).
- Pre-processed datasets write partitions in `<dataset>/partitions/`. Online datasets partition files (chunks of a file always belong to the same partition), and write partitions as json files referring to audio paths in `.ad_partitions/` of the first dataset folder (or in the `partitions` subfolder of `index_dir`, or in `partition_dir` if given) ; partitions are robust to added / removed files.
