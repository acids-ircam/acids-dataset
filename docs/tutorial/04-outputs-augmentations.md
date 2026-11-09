# 4. Outputs and augmentations

## Output patterns

Output patterns set the structure of items : a single field, a tuple, or a dict, with aliases (`->`) and slices.

```python
from acids_dataset.datasets import AudioDataset
from acids_dataset.features import Loudness, Chroma

dataset = AudioDataset("tutorial_dataset", sr=22050, chunk_length=1., channels=2,
                       features=[Loudness(sr=22050, keep_channels=False), Chroma(sr=22050)])

print(dataset.get(0, "waveform").shape)                         # a single tensor
x, l = dataset.get(0, "(waveform,loudness)")                    # a tuple
item = dataset.get(0, "{waveform->x,chroma->c}")                # a dict, with aliases
left_right = dataset.get(0, "{waveform[0]->left,waveform[1]->right}")
print(item["x"].shape, item["c"].shape, left_right["left"].shape)
```

The `index` entry gives the index of items (used to [record losses](07-training.md)) :

```python
print(dataset.get(3, "{waveform,index}")["index"])
```

## Augmentations

Transforms apply to outputs : one transform for single patterns, a list for tuples, a dict of aliases for dicts. Transforms of `acids_dataset.transforms` work on tensors and numpy arrays, and apply randomly with a probability `p` :

```python
from acids_dataset import transforms as adt

augment = adt.Compose(adt.Gain(sr=22050, p=0.8), adt.PolarityInversion(sr=22050, p=0.5))
dataset = AudioDataset("tutorial_dataset", sr=22050, chunk_length=1., features=[Loudness(sr=22050, keep_channels=False)],
                       output_pattern="{waveform,loudness}", transforms={"waveform": augment})
print(dataset[0]["waveform"].abs().max())
```

Features are computed **after** transforms : here, `loudness` is the loudness of the augmented waveform. To compute features on the original audio, before transforms, give `features_after_transforms=False` (to the dataset, `configure` or `get_loader`) :

```python
original = dataset.configure(features_after_transforms=False)
print(float(dataset[0]["loudness"]), float(original[0]["loudness"]))
```

Any callable can be a transform, e.g. a lambda :

```python
dataset.transforms = {"loudness": lambda l: (l + 70) / 70}     # normalized loudness
print(float(dataset[0]["loudness"]))
```

!!! tip
    Transforms run in data loader workers. With `spawn` workers (default on macOS), transforms must be picklable : prefer functions and classes defined in modules over lambdas.

Next : [partitions](05-partitions.md).

!!! info "See also"
    [Output patterns](../guide/output-patterns.md) and [Augmentations](../guide/augmentations.md) in the documentation.
