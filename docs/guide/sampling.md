# Balancing and curriculum

## Class balancing

`get_loader` / `get_sampler` can balance the labels of features : values of categorical features (`meta_regexp` features, `original_path`, hashed features), or histogram (quantile) bins of continuous features (e.g. `loudness`, with `n_bins`).

```python
loader = dataset.get_loader(batch_size=32, balance_by="instrument")                         # each instrument equally likely
loader = dataset.get_loader(batch_size=32, balance_by=["instrument", "pitch"], balance_power=0.5)
loader = dataset.get_loader(batch_size=32, balance_by="loudness", sampler_kwargs={"n_bins": 10})  # flat loudness distribution
loader = dataset.get_loader(balance_by="instrument", balanced_batches=(8, 4))                 # 8 instruments x 4 items per batch
```

- Item weights are `count(label) ** -balance_power` : `balance_power=1` samples labels uniformly, `0` samples items uniformly (no balancing), and values in between soften the imbalance. Items without label are excluded (or form their own class with `sampler_kwargs={"missing": "class"}`).
- In online mode, labels of path-based features are given by files, and continuous features are averaged over the items of each file. `dataset.item_labels(features)` gives the label of each item.
- In combined datasets, labels are balanced within each dataset, and datasets are sampled with their `weights`.
- `balanced_batches=(classes_per_batch, items_per_class)` gives batches of distinct labels, e.g. for contrastive learning.

Samplers are reproducible with `seed`, and distributed : each process gets its share of items, from `rank` / `world_size` (`sampler_kwargs`), or from `torch.distributed` if initialized. Samplers are `DistributedSampler` subclasses : Lightning does not wrap them, and calls their `set_epoch` at each epoch (with `balanced_batches` in distributed training, use `Trainer(use_distributed_sampler=False)`).

## Curriculum

A `Curriculum` schedules sampled items from their difficulty : items whose difficulty rank is below a competence `c(t)` are sampled, `c(t)` growing from `start` to `end` along `length` epochs or steps (`unit`), with a `"linear"`, `"sqrt"` or `"exp"` schedule. With `mode="soft"`, weights decrease smoothly above the competence. Curricula combine with class balancing (weights are multiplied).

```python
from acids_dataset.sampling import Curriculum, rarity

curriculum = Curriculum(rarity(dataset.item_labels("instrument")), start=0.3, end=1., length=20, unit="epoch")
loader = dataset.get_loader(batch_size=32, curriculum=curriculum, seed=0)
for epoch in range(n_epochs):
    loader.sampler.set_epoch(epoch)       # done by Lightning
    ...
```

Difficulties can be :
- the **rarity** of labels (`rarity(labels)` : `-log(frequency)`), such that common labels come first ;
- **custom difficulties** given by the user : an array with a difficulty for each item (e.g. from an external model, annotations, or precomputed descriptors), or a callable `difficulty(epoch, step)` returning such an array, e.g. :

```python
import numpy as np
difficulty = np.array([my_difficulty(dataset.get_fragment(...)) for ...])   # one value per item (see dataset.files / item_labels)
curriculum = Curriculum(difficulty, start=0.2, length=10)
curriculum = Curriculum(lambda epoch, step: scores_at(epoch), start=0.2, length=10)
```

Items with unknown difficulties (NaN) are ranked in the middle.
- **training losses**, recorded by a `LossTracker` (see below).

With `unit="step"`, items are drawn by blocks (`block_size`, 1024 by default) with the competence at the current step, given by `curriculum.step(step)` (done by the Lightning callback).

## Loss feedback

A `LossTracker` records the losses of items during training, and gives their difficulty to a curriculum (self-paced learning : items with low losses first). Losses are recorded per item with a strategy (`"ema"` exponential moving average with `momentum`, `"mean"`, `"last"`, or `"max"`) ; with `groups` (e.g. `dataset.item_labels("original_path")`), items that were never seen get the mean loss of their group (e.g. their file), and `tracker.statistics()` gives the distribution of losses among items and groups. Item indices are given by an `index` entry in the output pattern.

```python
import pytorch_lightning as pl
from acids_dataset.sampling import Curriculum, LossTracker
from acids_dataset.lightning import CurriculumCallback

dataset.output_pattern = "{waveform,index}"
tracker = LossTracker(len(dataset), strategy="ema", groups=dataset.item_labels("original_path"))
curriculum = Curriculum(tracker, start=0.3, length=20)
loader = dataset.get_loader(batch_size=32, curriculum=curriculum, seed=0)

class Model(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        losses = self.item_losses(batch["waveform"])          # one loss per item
        return {"loss": losses.mean(), "item_losses": losses.detach()}

trainer = pl.Trainer(callbacks=[CurriculumCallback(curriculum, tracker)])
trainer.fit(Model(), loader)
```

`CurriculumCallback` sets the steps of step-based curricula, records item losses (gathered over processes in distributed training), logs the competence and the mean recorded loss, and saves curricula and trackers in checkpoints (several callbacks need different `name`s). Without Lightning, call `tracker.update(batch["index"], losses)` and `curriculum.step()` / `sampler.set_epoch(epoch)` in the training loop.
