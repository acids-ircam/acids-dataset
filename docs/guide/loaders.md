# Data loaders

`get_loader` (available in both modes) returns a `torch.utils.data.DataLoader` with a random sampler (or a sequential one with `shuffle=False`). Arguments of `DataLoader` (`num_workers`, `drop_last`...) are given to the loader, other ones (`chunk_length`, `batch_by`, `pad_mode` in online mode) configure the dataset.

```python
loader = dataset.get_loader(batch_size=16, num_workers=4, drop_last=True)
loader = dataset.get_loader(batch_size=16, max_samples=10000, replacement=True) # fixed number of samples per epoch
```

The chunk length of pre-processed datasets is fixed at compilation : giving a different `chunk_length` to `get_loader` raises an error.

## Chunking strategy (online mode)

In online mode, dataset items are chunks of `chunk_length` samples, that can be given at initialization, or later with `configure` or `get_loader` (that return configured copies of the dataset). The way chunks are distributed in the dataset is set by `batch_by` :

| `batch_by`  | Items                                                                                                                                                                                                                                  |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"files"`   | one item per file, taken at a random position of the file. Files shorter than `chunk_length` are padded with `pad_mode`. Each file has the same weight, whatever its duration.                                                         |
| `"dataset"` | each file is split in `round(duration / chunk_length)` slots (at least 1), and each item is taken at a random position in its slot. The dataset length is then about `total_duration / chunk_length`, and chunks never overlap two files. |
| `"auto"`    | `"dataset"` if the median file duration is longer than `auto_ratio * chunk_length` (a few long files, default `auto_ratio=2`), `"files"` otherwise (a lot of short files).                                                              |

As items are taken at random positions, the same index gives different chunks across epochs (random positions are seeded independently in each `DataLoader` worker). `dataset.chunk_position(index, random_offset=False)` gives the deterministic (file, start) position of an item.

```python
dataset = AudioDataset("path/to/your/dataset", sr=44100)
train_loader = dataset.get_loader(batch_size=16, chunk_length=1., batch_by="dataset", num_workers=4)
len(train_loader.dataset)                    # ~ total duration / 1 second
dataset.configure(chunk_length=2.).batch_by  # "auto" is resolved once chunk_length is known
```

If files have different sampling rates, `sr` must be given so that all the items of a batch have the same size.

With `loudness_threshold` (in LUFS, e.g. `-60`), chunks quieter than the threshold (e.g. silence) are drawn again at a random position of their file, up to `max_retries` times (10 by default) ; files that are silent everywhere still give silent chunks. This is the online counterpart of the `loudness_threshold` of pre-processing (the `after` preset sets both).

Without `chunk_length`, items are whole files (loaded at `sr`, or at their own sampling rate if `sr` is not given). As files have different lengths, they cannot be batched : `get_loader` then requires `batch_size=1` (or a `collate_fn`).

## Batch features

Features computed dynamically (in online mode, or features that are not stored in a pre-processed dataset) can be computed on whole batches in the main process, e.g. on GPU, instead of item by item in data loader workers :

```python
dataset = AudioDataset("path/to/raw/audio", chunk_length=131072, features=[Loudness(), ModuleEmbedding(module_path="model.ts", name="z")],
                       output_pattern="{waveform->x,loudness,z}", transforms={"x": adt.Gain()})
loader = dataset.get_loader(batch_size=32, num_workers=8, batch_features=["z"], device="cuda")   # z is computed on cuda, on batches
loader = dataset.get_loader(batch_size=32, batch_features=["z"], batch_transforms=True, device="cuda")  # transforms on cuda too
```

- `batch_features` are names of features computed by the dataset, feature objects (e.g. to compute new features on pre-processed datasets, whose stored features are read), or `"auto"` (features on accelerators). Other features are still computed in workers.
- By default, features of online datasets, item by item or on batches, are computed on the waveform output **after its transforms** (here, on the audio with gain), and on the untransformed audio if the waveform is not in the output pattern. With `features_after_transforms=False` (in the dataset, `configure` or `get_loader`), they are computed on the loaded audio, **before** transforms (see [features and augmentations](augmentations.md#features-and-augmentations)). Transforms of features themselves are applied after their computation.
- With `batch_transforms=True`, features of the output pattern computed in workers would miss the augmentations : they are then computed on batches too (unless `features_after_transforms=False`).
- With `batch_transforms=True`, transforms are applied on batches on `device` in the main process (numpy-based transforms move batches to the CPU and back).
- `get_loader` then returns a `BatchFeatureLoader`, that behaves as a `DataLoader`. Workers do not compute batch features : features on GPU do not require spawned workers anymore.
- With custom data loaders, `dataset.compute_batch_features(audio, features=None, device=None)` computes features on a batch `(batch, channels, samples)`, and returns `{name: feature}`.

All features compute batches at once (`supports_batch`), except `BeatTrack`, `AfterMIDI` and `RegexpFeature`, that are computed item by item (with the same results), as well as `ModuleEmbedding` with transforms.

## Distributed training

Samplers given by `get_loader` / `get_sampler` with `seed`, `balance_by` or `curriculum` are `acids_dataset.sampling.WeightedItemSampler` objects :

- they are reproducible : with the same `seed`, every process draws the same items in the same order ;
- they are distributed : each process takes its share of items, from `rank` and `world_size` (given with `sampler_kwargs`, or read from `torch.distributed` if it is initialized) ;
- they are `DistributedSampler` subclasses : Lightning does not wrap them, and calls their `set_epoch` at each epoch.

```python
loader = dataset.get_loader(batch_size=16, seed=0, num_workers=4)                                  # reproducible, distributed
loader = dataset.get_loader(batch_size=16, seed=0, sampler_kwargs={"rank": rank, "world_size": 4})  # explicit processes
```

Partitions must also be identical across processes : give the same `seed` to `split` in every process, or write partitions in one process and load them in the others (see [Partitions](partitions.md)).

!!! warning "GPU features in workers"
    Features computed in data loader workers on GPU require workers started with `spawn` (CUDA cannot be used in forked processes), which `get_loader` sets automatically. Each worker then holds its own copy of models : prefer [batch features](#batch-features), computed in the main process.
