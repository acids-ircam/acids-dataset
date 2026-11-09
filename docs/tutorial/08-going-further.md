# 8. Going further

## Presets

Presets are gin files gathering parsing settings (sampling rate, channels, chunk length, features...), shared by online and pre-processed datasets :

```python
from acids_dataset.datasets import AudioDataset
from acids_dataset.presets import list_presets

print(list(list_presets().keys()))
dataset = AudioDataset("tutorial_dataset", preset="rave", sr=22050)   # explicit arguments override the preset
print(dataset.chunk_length, dataset.sr)
```

## Combining datasets

`CombinedAudioDataset` combines datasets (online or pre-processed), with sampling weights :

```python
from acids_dataset.datasets import CombinedAudioDataset

combined = CombinedAudioDataset([AudioDataset("tutorial_compiled"), AudioDataset("tutorial_dataset", sr=22050, chunk_length=1.)],
                                weights=[0.7, 0.3])
print(len(combined), next(iter(combined.get_loader(batch_size=4))).shape)
```

## The dataset browser

The browser is a web interface to listen to items, check augmentations, compute statistics, create partitions and check the health of datasets (requires `pip install acids-dataset[ui]`) :

```bash
acids-dataset ui --path tutorial_dataset
```

## Where next

- the [documentation](../guide/index.md) covers every topic in detail,
- the [API reference](../reference/index.md) describes classes and functions.
