# 7. Training

## Balancing classes

Our dataset has 6 piano files, 3 bass files and 2 drums files. `balance_by` samples classes uniformly :

```python
import collections
from acids_dataset.datasets import AudioDataset

dataset = AudioDataset("tutorial_dataset", sr=22050, chunk_length=1., meta_regexp=["**/{{instrument}}_*.wav"],
                       output_pattern="{waveform,index}")
labels = dataset.item_labels("instrument")

sampler = dataset.get_sampler(balance_by="instrument", seed=0, max_samples=3000)
print(collections.Counter(labels[i] for i in sampler))     # about 1000 items of each instrument

loader = dataset.get_loader(batch_size=8, balance_by="instrument", balance_power=0.5, seed=0)
```

`balance_power` softens balancing (`0` : natural frequencies, `1` : uniform classes).

## Curriculum

A `Curriculum` samples easy items first, from their difficulty. Here, rare instruments come last :

```python
from acids_dataset.sampling import Curriculum, rarity

curriculum = Curriculum(rarity(labels), start=0.5, end=1., length=3)
sampler = dataset.get_sampler(curriculum=curriculum, seed=0, max_samples=1000)
for epoch in range(4):
    sampler.set_epoch(epoch)
    print(epoch, sorted(set(labels[i] for i in sampler)))
```

Difficulties can also be given by you (an array, or a function of the epoch), or by training losses.

## Training with Lightning

A `LossTracker` records the losses of items, and `CurriculumCallback` connects it to Lightning : items with low losses are sampled first (self-paced learning).

```python
import torch
import pytorch_lightning as pl
from acids_dataset.sampling import LossTracker
from acids_dataset.lightning import CurriculumCallback

tracker = LossTracker(len(dataset), groups=dataset.item_labels("original_path"))
curriculum = Curriculum(tracker, start=0.5, length=2)
loader = dataset.get_loader(batch_size=4, curriculum=curriculum, seed=0)

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Conv1d(1, 1, 9, padding=4)

    def training_step(self, batch, batch_idx):
        x = batch["waveform"]
        losses = (self.layer(x) - x).pow(2).mean((-1, -2))         # one loss per item
        return {"loss": losses.mean(), "item_losses": losses.detach()}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

trainer = pl.Trainer(max_epochs=2, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False,
                     callbacks=[CurriculumCallback(curriculum, tracker)])
trainer.fit(Model(), loader)
print(tracker.statistics()["n_seen"], "items seen")
```

## Features on GPU

Features computed on the fly (e.g. neural embeddings) can be computed on whole batches on GPU, instead of item by item in workers, with `batch_features` :

```python
from acids_dataset.features import Loudness

dataset = AudioDataset("tutorial_dataset", sr=22050, chunk_length=1., features=[Loudness(sr=22050, keep_channels=False)],
                       output_pattern="{waveform,loudness}")
loader = dataset.get_loader(batch_size=4, batch_features=["loudness"], device="cpu")   # device="cuda" on GPU
print({k: tuple(v.shape) for k, v in next(iter(loader)).items()})
```

Next : [going further](08-going-further.md).

!!! info "See also"
    [Balancing and curriculum](../guide/sampling.md) and [Data loaders](../guide/loaders.md) in the documentation.
