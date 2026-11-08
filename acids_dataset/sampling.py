"""Balanced and curriculum sampling.

    - balance_weights : sampling weights of items from their labels (classes of categorical features, or histogram bins of
      continuous features), such that classes are sampled uniformly (balance_power=1), with their natural frequency
      (balance_power=0), or in between.
    - WeightedItemSampler : weighted sampler of items, reproducible (seed, set_epoch) and distributed (rank, world_size),
      whose weights can follow a Curriculum.
    - Curriculum : schedule (in epochs or steps) of the sampled items, from their difficulty : items whose difficulty rank is
      below a competence c(t) are sampled, c(t) growing from start to end. Difficulties are given by the user (arrays or
      callables), by the rarity of feature values (rarity), by feature values (feature_difficulty), or by training
      losses (LossTracker).
    - LossTracker : records losses of items during training (several strategies), and gives their difficulty.
    - BalancedBatchSampler : batches of classes_per_batch classes with items_per_class items each.

See acids_dataset.lightning for a Lightning callback updating curricula and loss trackers."""
import math
import random
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler


def _distributed_info(rank: int | None, world_size: int | None):
    if rank is not None and world_size is not None:
        return int(rank), int(world_size)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return (torch.distributed.get_rank() if rank is None else rank,
                torch.distributed.get_world_size() if world_size is None else world_size)
    return (rank or 0), (world_size or 1)


# weights from labels

def label_counts(labels: Sequence[Any]) -> Dict[Any, int]:
    return dict(Counter(l for l in labels if l is not None))


def balance_weights(labels: Sequence[Any], power: float = 1., missing: str = "exclude") -> torch.Tensor:
    """sampling weights of items from their labels : w = count(label) ** -power, normalized (power=1 samples labels
    uniformly, power=0 samples items uniformly). Items without label (None) are excluded (missing="exclude") or form
    their own class (missing="class")."""
    assert missing in ["exclude", "class"], "missing must be exclude or class"
    if missing == "class":
        labels = ["__missing__" if l is None else l for l in labels]
    counts = label_counts(labels)
    weights = torch.tensor([0. if l is None else counts[l] ** (-float(power)) for l in labels], dtype=torch.float64)
    if weights.sum() <= 0:
        raise ValueError("no item has a label")
    return weights / weights.sum()


def bin_values(values: Sequence[float], n_bins: int = 10) -> List[Any]:
    """quantile bins (histogram) of continuous values, as labels ; non-finite values have no label (None)."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    labels = [None] * len(values)
    if finite.sum() == 0:
        return labels
    edges = np.unique(np.quantile(values[finite], np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return [0 if f else None for f in finite]
    bins = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    return [int(b) if f else None for b, f in zip(bins, finite)]


# difficulties

def rarity(labels: Sequence[Any]) -> torch.Tensor:
    """difficulty of items from the rarity of their labels : -log(frequency). Items without label get the highest rarity."""
    counts = label_counts(labels)
    n = max(1, sum(counts.values()))
    values = [math.log(n / counts[l]) if l is not None else float("nan") for l in labels]
    out = torch.tensor(values, dtype=torch.float64)
    if torch.isnan(out).any():
        out[torch.isnan(out)] = out[~torch.isnan(out)].max() if (~torch.isnan(out)).any() else 0.
    return out


def _rank_quantiles(difficulty: torch.Tensor) -> torch.Tensor:
    """quantiles of difficulties in [0, 1] (average ranks for ties ; unknown difficulties are considered as average)."""
    full = torch.as_tensor(difficulty, dtype=torch.float64)
    unknown = ~torch.isfinite(full)
    quantiles = torch.full_like(full, 0.5)
    if unknown.all():
        return quantiles
    d = full[~unknown]
    order = torch.argsort(d, stable=True)
    ranks = torch.empty_like(d)
    ranks[order] = torch.arange(len(d), dtype=torch.float64)
    # average ranks of ties
    values, inverse = torch.unique(d, return_inverse=True)
    sums = torch.zeros(len(values), dtype=torch.float64).index_add_(0, inverse, ranks)
    counts = torch.zeros(len(values), dtype=torch.float64).index_add_(0, inverse, torch.ones_like(ranks))
    ranks = (sums / counts)[inverse]
    quantiles[~unknown] = ranks / max(1, len(d) - 1)
    return quantiles


class LossTracker():
    """records losses of items during training, and gives their difficulty.

    Args:
        n_items (int): number of items of the dataset.
        strategy (str): how losses of an item are recorded : "ema" (exponential moving average, with momentum), "mean"
            (running mean), "last" (last loss), or "max" (maximum loss).
        momentum (float): momentum of "ema".
        groups (Sequence | None): group of each item (e.g. its file, or its class). Items that were never seen get the mean
            difficulty of their group (or the global mean), and statistics are given by group.
    """
    strategies = ["ema", "mean", "last", "max"]

    def __init__(self, n_items: int, strategy: str = "ema", momentum: float = 0.9, groups: Sequence[Any] | None = None):
        assert strategy in self.strategies, "strategy must be in %s"%self.strategies
        self.n_items = int(n_items)
        self.strategy = strategy
        self.momentum = momentum
        self.values = torch.zeros(self.n_items, dtype=torch.float64)
        self.counts = torch.zeros(self.n_items, dtype=torch.long)
        self.groups = None if groups is None else list(groups)
        if self.groups is not None:
            assert len(self.groups) == self.n_items, "groups must give a group for each item"

    def update(self, indices, losses) -> None:
        """records losses of items (indices and losses are sequences or tensors of the same length)."""
        indices = torch.as_tensor(indices).reshape(-1).long().cpu()
        losses = torch.as_tensor(losses).detach().reshape(-1).double().cpu()
        assert len(indices) == len(losses), "indices and losses must have the same length"
        for i, l in zip(indices.tolist(), losses.tolist()):
            if not math.isfinite(l): continue
            n = self.counts[i].item()
            if n == 0 or self.strategy == "last":
                self.values[i] = l
            elif self.strategy == "ema":
                self.values[i] = self.momentum * self.values[i] + (1 - self.momentum) * l
            elif self.strategy == "mean":
                self.values[i] = self.values[i] + (l - self.values[i]) / (n + 1)
            elif self.strategy == "max":
                self.values[i] = max(self.values[i].item(), l)
            self.counts[i] += 1

    @property
    def seen(self) -> torch.Tensor:
        return self.counts > 0

    def difficulty(self) -> torch.Tensor:
        """recorded losses of items ; unseen items get the mean loss of their group (or the global mean, or nan if no item
        was seen)."""
        out = self.values.clone()
        seen = self.seen
        if not seen.any():
            return torch.full_like(out, float("nan"))
        global_mean = out[seen].mean()
        out[~seen] = global_mean
        if self.groups is not None:
            means = self.group_means()
            for i in torch.where(~seen)[0].tolist():
                out[i] = means.get(self.groups[i], global_mean.item())
        return out

    def group_means(self) -> Dict[Any, float]:
        assert self.groups is not None, "LossTracker has no groups"
        sums, counts = {}, {}
        for i in torch.where(self.seen)[0].tolist():
            g = self.groups[i]
            sums[g] = sums.get(g, 0.) + self.values[i].item()
            counts[g] = counts.get(g, 0) + 1
        return {g: sums[g] / counts[g] for g in sums}

    def statistics(self) -> Dict[str, Any]:
        """distribution of recorded losses among items."""
        seen = self.seen
        stats = {'n_seen': int(seen.sum()), 'ratio_seen': float(seen.float().mean())}
        if seen.any():
            values = self.values[seen]
            stats.update({'mean': float(values.mean()), 'std': float(values.std()) if len(values) > 1 else 0.,
                          'quantiles': {str(q): float(torch.quantile(values, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}})
            if self.groups is not None:
                stats['groups'] = self.group_means()
        return stats

    def state_dict(self) -> Dict[str, Any]:
        return {'values': self.values.clone(), 'counts': self.counts.clone(), 'strategy': self.strategy, 'momentum': self.momentum}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.values, self.counts = state['values'].clone(), state['counts'].clone()
        self.strategy, self.momentum = state['strategy'], state['momentum']


class Curriculum():
    """schedule of sampled items from their difficulty (see module documentation).

    Args:
        difficulty: difficulty of each item, as an array, a callable(epoch, step) returning an array, or a LossTracker.
        start (float): competence at the beginning of the schedule (ratio of easiest items that are sampled).
        end (float): competence at the end of the schedule.
        length (int): duration of the schedule, in epochs or steps (see unit).
        unit (str): "epoch" or "step". Epochs are given by set_epoch (called by samplers and Lightning), steps by step.
        schedule (str): "linear", "sqrt" (fast growth then slow), or "exp" (slow growth then fast).
        mode (str): "threshold" (only items below the competence are sampled) or "soft" (weights decrease smoothly above
            the competence, with temperature).
        temperature (float): temperature of "soft" mode.
    """
    schedules = ["linear", "sqrt", "exp"]

    def __init__(self, difficulty: Sequence[float] | Callable[[int, int], Sequence[float]] | "LossTracker", start: float = 0.2, end: float = 1., length: int = 10, unit: str = "epoch",
                 schedule: str = "linear", mode: str = "threshold", temperature: float = 0.05):
        assert unit in ["epoch", "step"], "unit must be epoch or step"
        assert schedule in self.schedules, "schedule must be in %s"%self.schedules
        assert mode in ["threshold", "soft"], "mode must be threshold or soft"
        assert 0 < start <= 1 and 0 < end <= 1, "competences must be in ]0, 1]"
        self.difficulty = difficulty
        self.start, self.end, self.length = float(start), float(end), max(1, int(length))
        self.unit, self.schedule, self.mode, self.temperature = unit, schedule, mode, temperature
        self.current_epoch = 0
        self.current_step = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def step(self, step: int | None = None) -> None:
        """sets the current step (or increments it if step is None)."""
        self.current_step = self.current_step + 1 if step is None else int(step)

    @property
    def progress(self) -> float:
        t = self.current_epoch if self.unit == "epoch" else self.current_step
        return min(1., max(0., t / self.length))

    @property
    def competence(self) -> float:
        p = self.progress
        if self.schedule == "sqrt":
            p = math.sqrt(p)
        elif self.schedule == "exp":
            p = (math.exp(3 * p) - 1) / (math.exp(3) - 1)
        return self.start + (self.end - self.start) * p

    def _difficulty(self) -> torch.Tensor:
        d = self.difficulty
        if isinstance(d, LossTracker):
            return d.difficulty()
        if callable(d):
            d = d(self.current_epoch, self.current_step)
        return torch.as_tensor(np.asarray(d, dtype=np.float64))

    def weights(self) -> torch.Tensor:
        """sampling weights of items at the current epoch / step (not normalized)."""
        quantiles = _rank_quantiles(self._difficulty())
        c = self.competence
        if self.mode == "soft":
            return torch.sigmoid((c - quantiles) / self.temperature)
        weights = (quantiles <= c + 1e-9).double()
        if weights.sum() == 0:
            weights = (quantiles <= quantiles.min() + 1e-9).double()
        return weights

    def state_dict(self) -> Dict[str, Any]:
        state = {'current_epoch': self.current_epoch, 'current_step': self.current_step}
        if isinstance(self.difficulty, LossTracker):
            state['tracker'] = self.difficulty.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.current_epoch, self.current_step = state['current_epoch'], state['current_step']
        if 'tracker' in state and isinstance(self.difficulty, LossTracker):
            self.difficulty.load_state_dict(state['tracker'])


class WeightedItemSampler(DistributedSampler):
    """weighted sampler of dataset items, reproducible and distributed.

    Each epoch, num_samples * world_size items are drawn with a generator seeded by (seed, epoch), identical on every
    process, and each process takes its share (rank::world_size). Weights are static weights (e.g. balance_weights),
    multiplied by the weights of a curriculum. If the curriculum is step-based, items are drawn by blocks of block_size,
    with the weights of the curriculum at the current step. Subclass of DistributedSampler, such that Lightning does not
    wrap it, and calls set_epoch at each epoch.
    """

    def __init__(self, n_items: int, weights=None, curriculum: Curriculum | None = None, num_samples: int | None = None,
                 replacement: bool = True, seed: int | None = None, rank: int | None = None, world_size: int | None = None,
                 block_size: int | None = None):
        self.n_items = int(n_items)
        self.static_weights = None if weights is None else torch.as_tensor(weights, dtype=torch.float64)
        if self.static_weights is not None:
            assert len(self.static_weights) == self.n_items, "weights must have a weight for each item"
        self.curriculum = curriculum
        self.rank, self.num_replicas = _distributed_info(rank, world_size)
        self.num_samples = int(num_samples) if num_samples is not None else int(math.ceil(self.n_items / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.seed = random.randrange(2 ** 31) if seed is None else int(seed)
        self.block_size = block_size if block_size is not None else (1024 if curriculum is not None and curriculum.unit == "step" else None)
        self.epoch = 0
        self.shuffle, self.drop_last, self.dataset = True, False, None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if self.curriculum is not None:
            self.curriculum.set_epoch(epoch)

    def weights(self) -> torch.Tensor:
        weights = torch.ones(self.n_items, dtype=torch.float64) if self.static_weights is None else self.static_weights.clone()
        if self.curriculum is not None:
            curriculum_weights = self.curriculum.weights()
            assert len(curriculum_weights) == self.n_items, "curriculum difficulties must be given for each item"
            weights = weights * curriculum_weights
        if weights.sum() <= 0:
            raise ValueError("all sampling weights are zero")
        return weights

    def _draw(self, n: int, generator) -> List[int]:
        weights = self.weights()
        replacement = self.replacement or n > int((weights > 0).sum())
        return torch.multinomial(weights, n, replacement=replacement, generator=generator).tolist()

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + 1_000_003 * self.epoch)
        block = self.block_size or self.total_size
        block = max(self.num_replicas, (block // self.num_replicas) * self.num_replicas)
        drawn = 0
        while drawn < self.total_size:
            n = min(block, self.total_size - drawn)
            # n is a multiple of num_replicas (except for the last block), such that shares stay balanced
            indices = self._draw(n, generator)
            yield from indices[self.rank::self.num_replicas]
            drawn += n

    def __len__(self) -> int:
        return self.num_samples


class BalancedBatchSampler():
    """batches of classes_per_batch distinct classes, with items_per_class items of each class (drawn with replacement for
    classes with fewer items). Batches are drawn with a generator seeded by (seed, epoch), and distributed among
    processes (rank::world_size)."""

    def __init__(self, labels: Sequence[Any], classes_per_batch: int, items_per_class: int, num_batches: int | None = None,
                 seed: int | None = None, rank: int | None = None, world_size: int | None = None):
        self.classes = {}
        for i, l in enumerate(labels):
            if l is not None:
                self.classes.setdefault(l, []).append(i)
        self.class_names = list(self.classes.keys())
        if classes_per_batch > len(self.class_names):
            raise ValueError("classes_per_batch (%d) exceeds the number of classes (%d)"%(classes_per_batch, len(self.class_names)))
        self.classes_per_batch, self.items_per_class = int(classes_per_batch), int(items_per_class)
        self.rank, self.num_replicas = _distributed_info(rank, world_size)
        n_items = sum(len(v) for v in self.classes.values())
        self.num_batches = int(num_batches) if num_batches is not None else max(1, n_items // (self.batch_size * self.num_replicas))
        self.seed = random.randrange(2 ** 31) if seed is None else int(seed)
        self.epoch = 0
        # Lightning calls set_epoch of batch_sampler.sampler
        self.sampler = self

    @property
    def batch_size(self) -> int:
        return self.classes_per_batch * self.items_per_class

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        for b in range(self.num_batches * self.num_replicas):
            classes = rng.sample(self.class_names, self.classes_per_batch)
            batch = []
            for c in classes:
                items = self.classes[c]
                batch.extend(rng.sample(items, self.items_per_class) if len(items) >= self.items_per_class
                             else rng.choices(items, k=self.items_per_class))
            if b % self.num_replicas == self.rank:
                yield batch

    def __len__(self) -> int:
        return self.num_batches


__all__ = ['balance_weights', 'bin_values', 'rarity', 'LossTracker', 'Curriculum', 'WeightedItemSampler', 'BalancedBatchSampler']
