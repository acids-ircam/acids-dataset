# acids-dataset

!!! warning
    `acids-dataset` is still experimental, do not hesitate to [open issues](https://github.com/acids-ircam/acids-dataset/issues)!

`acids-dataset` prepares audio data for generative models, such as [RAVE](https://github.com/acids-ircam/RAVE) and [AFTER](https://github.com/acids-ircam/AFTER). It loads audio folders and gives PyTorch datasets and data loaders, with :

- **two modes** sharing the same interface : *online* datasets, that read raw audio folders directly, and *pre-processed* datasets, compiled in an [LMDB](https://openldap.org/) database for fast training ;
- **features** computed before-hand or on the fly : loudness, pitch, spectral, rhythmic, harmonic and spatial descriptors, neural embeddings, metadata from file names... ;
- **flexible outputs** (tuples, dicts, slices) and **augmentations** with probabilities ;
- **partitions** (random, by file, by class, stratified over continuous features), **class balancing** and **curriculum learning**, compatible with distributed training and Lightning ;
- a **dataset browser** and **health reports** to explore and check datasets.

## Where to start

<div class="grid cards" markdown>

- **[Step by step](tutorial/index.md)**

    A tutorial that builds a small dataset and goes through the main features, from loading audio to training with Lightning.

- **[Documentation](guide/index.md)**

    The full user guide, by topic : datasets, loaders, features, partitions, sampling, pre-processing, presets, command line...

- **[API reference](reference/index.md)**

    Classes and functions, generated from the code.

</div>

## In a nutshell

```python
from acids_dataset.datasets import AudioDataset
from acids_dataset.features import Loudness

dataset = AudioDataset("path/to/audio", sr=44100, chunk_length=131072, features=[Loudness()],
                       output_pattern="{waveform,loudness}")
partitions = dataset.split({"train": 0.9, "valid": 0.1}, features=["original_path"], seed=0)
loader = partitions["train"].get_loader(batch_size=16, num_workers=4)
batch = next(iter(loader))     # {"waveform": (16, 1, 131072), "loudness": (16,)}
```

```bash
pip install acids-dataset
acids-dataset preprocess --path path/to/audio --out path/to/compiled --preset rave
acids-dataset ui --path path/to/compiled
```
