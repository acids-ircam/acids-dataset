# acids-dataset

[![Documentation](https://app.readthedocs.org/projects/acids-dataset/badge/?version=latest)](https://acids-dataset.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

> **Warning** `acids-dataset` is still experimental, do not hesitate to [open issues](https://github.com/acids-ircam/acids-dataset/issues)!

`acids-dataset` prepares audio data for generative models, such as [RAVE](https://github.com/acids-ircam/RAVE) and [AFTER](https://github.com/acids-ircam/AFTER), and is open to custom use. It loads audio folders and gives PyTorch datasets and data loaders, with :

- **two modes sharing the same interface** : *online* datasets, that read raw audio folders directly (no compilation step), and *pre-processed* datasets, compiled in an [LMDB](https://openldap.org/) database for fast training ;
- **features**, computed before-hand or on the fly : loudness, pitch, spectral, rhythmic, harmonic and spatial descriptors, neural embeddings, metadata extracted from file names... ;
- **flexible outputs** (tuples, dicts, slices) and **audio augmentations** with probabilities, including spatial / ambisonic transforms ;
- **partitions** (random, by file, by class, stratified over continuous features), **class balancing**, **multi-dataset weighting** and **curriculum learning**, compatible with distributed training and Lightning ;
- **TorchScript modules** of features and transforms, to reuse them at inference time ;
- a **dataset browser** and **health reports** to explore and check datasets ;
- **presets** to share parsing settings between datasets, scripts and the command line.

## Installation

```bash
pip install acids-dataset
```

Python 3.10 or above is required. Optional dependencies :

| Extra | Installs | For |
| ----- | -------- | --- |
| `acids-dataset[ui]` | [gradio](https://www.gradio.app), matplotlib | the dataset browser |
| `acids-dataset[pesto]`, `[crepe]`, `[pyworld]` | [pesto-pitch](https://github.com/SonyCSLParis/pesto), [torchcrepe](https://github.com/maxrmorrison/torchcrepe), [pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) | additional pitch / F0 estimators (`yin` and `pyin` are available through `librosa`) |

Audio is loaded with [soundfile](https://github.com/bastibe/python-soundfile) (`wav`, `flac`, `aiff`, `ogg`, `opus`, `mp3`, with sample-accurate partial loading), and with [torchcodec](https://github.com/pytorch/torchcodec) (FFmpeg) for other formats (`m4a`, `aac`...).

## Quick start

```python
from acids_dataset.datasets import AudioDataset
from acids_dataset.features import Loudness

# a raw folder gives an online dataset ; a pre-processed folder gives a pre-processed one
dataset = AudioDataset("path/to/audio", sr=44100, chunk_length=131072, features=[Loudness()],
                       output_pattern="{waveform,loudness}")
partitions = dataset.split({"train": 0.9, "valid": 0.1}, features=["original_path"], seed=0)
loader = partitions["train"].get_loader(batch_size=16, num_workers=4)
batch = next(iter(loader))     # {"waveform": (16, 1, 131072), "loudness": (16,)}
```

To compile a dataset (and use it exactly the same way) :

```bash
acids-dataset preprocess --path path/to/audio --out path/to/compiled --preset rave --feature loudness
acids-dataset info --path path/to/compiled --health      # describe the dataset, and check its health
acids-dataset ui --path path/to/compiled                 # explore it in the browser
```

The command line also provides `update` (add features or files to a compiled dataset), `add_embedding` (embeddings of a TorchScript model) and `presets`. `acids-dataset <command> --help` lists the options of each command.

## Documentation

The full documentation is on [Read the Docs](https://acids-dataset.readthedocs.io/en/latest/). It is written in `docs/`, with :

- a **[step by step tutorial](docs/tutorial/index.md)**, from loading audio to training with Lightning ;
- the **[user guide](docs/guide/index.md)**, by topic : datasets, loaders, features, augmentations, partitions, sampling, pre-processing, presets, command line, browser, extending the package ;
- the **API reference**, generated from the code.

Additional tutorials are available as Jupyter notebooks in `extras/`.

To build the documentation locally :

```bash
pip install -e ".[docs]"    # or pip install -r docs/requirements.txt
mkdocs serve                # local preview at http://127.0.0.1:8000
mkdocs build --strict       # static site in site/
```

## Development

```bash
pip install -e ".[ui]" pytest pytest-timeout gradio_client
pytest tests -m "not slow"     # fast suite (~10 minutes)
pytest tests -m slow           # long parametrized grids
```

`tests/test_transforms.py` has known failures (see `TODO.md`) and is not run in continuous integration. CI (`.github/workflows/tests.yml`) checks Python 3.10 compatibility with [vermin](https://github.com/netromdk/vermin), runs the fast suite on Python 3.10 to 3.13, and on Python 3.10 with the minimum versions of `numpy`, `scipy`, `scikit-learn` and `librosa` declared in `pyproject.toml`. See the [development guide](docs/guide/development.md) for details.

## Contributors

Axel Chemla--Romeu-Santos and Nils Demerlé, at [IRCAM](https://www.ircam.fr/) ([ACIDS](https://acids.ircam.fr/) team).

## License

[Creative Commons Attribution-NonCommercial 4.0 International](LICENSE).
