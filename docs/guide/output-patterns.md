# Output patterns

Outputs of datasets are set by an output pattern, that allows to index / transform complex datasets, and provide easy way to perform multi-view indexing in a very simple way.

```python
import torch
from acids_dataset import transforms as adt
from acids_dataset.datasets import AudioDataset

data_path = "..."
one_hot = lambda p: torch.nn.functional.one_hot(p.long() % 12, 12) # transforms can be any callable
dataset = AudioDataset(data_path)
dataset[0] # will provide automatically the "waveform" field as a single tensor
dataset.get(0) # __getitem__ is a proxy for .get function, that's actually much more powerful
dataset.get(0, "(waveform,mel)") # => returns both waveform and mel fields as a tuple
dataset.get(0, "{waveform,mel}") # => returns both waveform and mel fields as a dictionary with waveform and mel keys
dataset.get(0, "{waveform,mel->z}") # => same, but put mel vector in field "z"
dataset.get(0, "{waveform[0]->x_l, waveform[1]->x_r}") # => puts first channel in x_l field, second in x_r field
dataset.get(0, "(waveform,pitch)", [adt.Gain(), one_hot]) # => you can add also transforms (one per output, see below)
dataset.get(0, "{waveform,pitch}", {'pitch': one_hot}) # => same with dict outputs, only for given keys

# you can also specify during initialisation
dataset = AudioDataset(data_path, output_pattern="{waveform->x,pitch->y}", transforms={"y": one_hot})
out = dataset[0] 
```

Datasets can also be indexed with slices or sequences of indices, that return batches of items (collated as with a `torch.utils.data.DataLoader`, or as a list if items have different shapes) :

```python
dataset[0:4]                                    # => tensor of shape (4, channels, chunk_length)
dataset[-2:]                                    # => last two items
dataset[[0, 5, 7]]                              # => lists, tuples, numpy arrays or tensors of indices
dataset.get(slice(0, 4), "{waveform,mel}")      # => dictionary of batched waveform and mel
```

Output patterns access the buffers of data fragments (waveform and array features). Path-based metadata extracted with `meta_regexp` is stored as fragment metadata, and is not accessible through output patterns.
