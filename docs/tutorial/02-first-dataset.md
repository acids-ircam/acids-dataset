# 2. A first dataset

`AudioDataset` loads an audio folder. Given a raw folder, it returns an *online* dataset, that reads audio on the fly (see [pre-processing](06-preprocessing.md) for compiled datasets).

```python
from acids_dataset.datasets import AudioDataset

dataset = AudioDataset("tutorial_dataset", sr=22050, chunk_length=1.)
print(type(dataset).__name__, len(dataset))
```

At creation, files are scanned (duration, sampling rate, channels), and this information is cached in a `.ad` index file in the folder, such that next loads are instantaneous.

## Items

Items are chunks of `chunk_length` samples (an `int`) or seconds (a `float`). By default, they are waveforms of shape `(channels, samples)` :

```python
x = dataset[0]
print(x.shape)                # torch.Size([1, 22050])
print(dataset[0:4].shape)     # slices give batches : torch.Size([4, 1, 22050])
```

How chunks are distributed in the dataset is set by `batch_by` : `"files"` gives one random chunk per file, `"dataset"` splits files in about `duration / chunk_length` slots (here, 3 per file), and `"auto"` (default) chooses from the durations of files.

```python
print(dataset.batch_by, len(dataset))
per_file = dataset.configure(batch_by="files")     # configure returns a configured copy
print(len(per_file))                                # one item per file
```

Chunks are taken at random positions in their slots : the same index gives different chunks across epochs.

## Files

The files of the dataset are listed in `dataset.files`, and audio can be loaded from any file :

```python
print(dataset.files[0]["audio_path"], dataset.files[0]["duration"])
wav = dataset.load_audio(0, start=0.5, length=1.)  # 1 second from 0.5 s
print(wav.shape)
```

## Data loaders

`get_loader` gives a PyTorch `DataLoader`. Loading arguments (`chunk_length`, `batch_by`...) can also be given here :

```python
loader = dataset.get_loader(batch_size=4, chunk_length=0.5)
batch = next(iter(loader))
print(batch.shape)            # torch.Size([4, 1, 11025])
```

Arguments of `DataLoader` (`num_workers`, `drop_last`...) are passed to it. Next : [computing features](03-features.md).

!!! info "See also"
    [Datasets](../guide/datasets.md) and [Data loaders](../guide/loaders.md) in the documentation.
