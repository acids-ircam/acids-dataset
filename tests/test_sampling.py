"""Balanced and curriculum sampling : weights from labels, distributed weighted sampler, curricula, loss trackers,
balanced batches, item indices in output patterns, and Lightning integration."""
import collections
import math
import pytest
import torch
import torchaudio

from acids_dataset import preprocess_dataset
from acids_dataset.datasets import AudioDataset, CombinedAudioDataset
from acids_dataset.features import Loudness
from acids_dataset.sampling import (balance_weights, bin_values, rarity, LossTracker, Curriculum, WeightedItemSampler,
                                    BalancedBatchSampler, _rank_quantiles)

SR = 22050
COUNTS = {"piano": 18, "guitar": 5, "drums": 1}
REGEXP = ["**/{{instrument}}_*.wav"]


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    root = tmp_path_factory.mktemp("sampling")
    g = torch.Generator().manual_seed(0)
    for inst, n in COUNTS.items():
        for i in range(n):
            gain = 0.02 + 0.3 * i / max(1, n)
            (root / "raw").mkdir(exist_ok=True)
            torchaudio.save(str(root / "raw" / f"{inst}_{i}.wav"), gain * torch.randn(1, SR, generator=g), sample_rate=SR)
    preprocess_dataset(root / "raw", out=root / "pre", chunk_length=SR, sample_rate=SR, channels=1, meta_regexp=REGEXP,
                       features=[Loudness(sr=SR, keep_channels=False)], force=True)
    return {"online": root / "raw", "preprocessed": root / "pre"}


def load(paths, mode, **kwargs):
    if mode == "online":
        return AudioDataset(paths["online"], sr=SR, chunk_length=SR, meta_regexp=REGEXP, features=[Loudness(sr=SR, keep_channels=False)],
                            write_index=False, **kwargs)
    return AudioDataset(paths["preprocessed"], **kwargs)


def frequencies(labels, sampler):
    counts = collections.Counter(labels[i] for i in sampler)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


# weights from labels

def test_balance_weights():
    labels = ["a"] * 8 + ["b"] * 2 + [None]
    w = balance_weights(labels, power=1.)
    assert w[-1] == 0 and abs(float(w[:8].sum()) - 0.5) < 1e-9
    w = balance_weights(labels, power=0.)
    assert torch.allclose(w[:10], torch.full((10,), 0.1, dtype=torch.float64))
    w = balance_weights(labels, power=1., missing="class")
    assert abs(float(w[-1]) - 1 / 3) < 1e-9
    assert bin_values([1., 2., 3., 4., float("nan")], n_bins=2) == [0, 0, 1, 1, None]


@pytest.mark.parametrize("mode", ["online", "preprocessed"])
def test_item_labels(paths, mode):
    dataset = load(paths, mode)
    labels = dataset.item_labels("instrument")
    assert len(labels) == len(dataset) and collections.Counter(labels) == collections.Counter({k: v for k, v in COUNTS.items()})
    # labels follow items
    for i in range(len(dataset)):
        assert labels[i] in dataset._fragment_from_index(i).get_metadata()['audio_path']
    binned = dataset.item_labels("loudness", n_bins=4)
    assert set(binned) == {0, 1, 2, 3}
    combined_labels = dataset.item_labels(["instrument", "loudness"], n_bins=2)
    assert all(isinstance(l, tuple) for l in combined_labels)
    with pytest.raises(ValueError):
        dataset.item_labels("unknown")


@pytest.mark.parametrize("mode", ["online", "preprocessed"])
@pytest.mark.parametrize("power", [0., 0.5, 1.])
def test_balanced_sampling(paths, mode, power):
    dataset = load(paths, mode)
    labels = dataset.item_labels("instrument")
    sampler = dataset.get_sampler(balance_by="instrument", balance_power=power, seed=0, max_samples=8000)
    weights = {k: v ** (1 - power) for k, v in COUNTS.items()}
    expected = {k: w / sum(weights.values()) for k, w in weights.items()}
    observed = frequencies(labels, sampler)
    for k in COUNTS:
        assert abs(observed[k] - expected[k]) < 0.025, (k, observed, expected)


def test_continuous_balancing(paths):
    dataset = load(paths, "preprocessed")
    bins = dataset.item_labels("loudness", n_bins=4)
    observed = frequencies(bins, dataset.get_sampler(balance_by="loudness", n_bins=4, seed=0, max_samples=8000))
    assert all(abs(v - 0.25) < 0.025 for v in observed.values())


def test_combined_balancing(paths):
    combined = CombinedAudioDataset([load(paths, "preprocessed"), load(paths, "online")], weights=[0.8, 0.2])
    labels = combined.item_labels("instrument")
    assert labels[0][0] == 0 and labels[-1][0] == 1
    observed = frequencies(labels, combined.get_sampler(balance_by="instrument", seed=0, max_samples=8000))
    for inst in COUNTS:
        assert abs(observed[(0, inst)] - 0.8 / 3) < 0.03 and abs(observed[(1, inst)] - 0.2 / 3) < 0.03
    # without balancing, datasets keep their weights
    n0 = len(combined.datasets[0])
    first = sum(1 for i in combined.get_sampler(seed=0, max_samples=4000) if i < n0) / 4000
    assert abs(first - 0.8) < 0.03


# sampler

def test_weighted_sampler_reproducible():
    weights = torch.rand(100, generator=torch.Generator().manual_seed(0))
    a, b = list(WeightedItemSampler(100, weights, seed=3)), list(WeightedItemSampler(100, weights, seed=3))
    assert a == b and len(a) == 100
    sampler = WeightedItemSampler(100, weights, seed=3)
    sampler.set_epoch(1)
    assert list(sampler) != a
    unique = list(WeightedItemSampler(100, None, replacement=False, seed=0))
    assert sorted(unique) == list(range(100))


def test_weighted_sampler_distributed():
    full = list(WeightedItemSampler(100, None, replacement=False, seed=5, rank=0, world_size=1))
    shards = [list(WeightedItemSampler(100, None, replacement=False, seed=5, rank=r, world_size=4)) for r in range(4)]
    assert all(len(s) == 25 for s in shards)
    assert sorted(sum(shards, [])) == list(range(100))
    assert [shards[r][0] for r in range(4)] == full[:4]
    # samplers are DistributedSampler subclasses : Lightning does not wrap them
    assert isinstance(WeightedItemSampler(100, seed=0), torch.utils.data.distributed.DistributedSampler)


# curriculum

def test_competence_schedules():
    for schedule in ["linear", "sqrt", "exp"]:
        c = Curriculum(torch.arange(10.), start=0.2, end=1., length=4, schedule=schedule)
        values = []
        for e in range(6):
            c.set_epoch(e)
            values.append(c.competence)
        assert values[0] == pytest.approx(0.2) and values[4] == pytest.approx(1.) and values[5] == pytest.approx(1.)
        assert values == sorted(values)
    sqrt, lin = Curriculum([0.], length=4, schedule="sqrt"), Curriculum([0.], length=4)
    sqrt.set_epoch(1); lin.set_epoch(1)
    assert sqrt.competence > lin.competence


def test_curriculum_weights():
    difficulty = torch.arange(10.)
    c = Curriculum(difficulty, start=0.3, length=1)
    assert c.weights().tolist() == [1.] * 3 + [0.] * 7
    c.set_epoch(1)
    assert c.weights().sum() == 10
    soft = Curriculum(difficulty, start=0.3, length=1, mode="soft", temperature=0.05)
    w = soft.weights()
    assert w[0] > 0.99 and w[-1] < 0.01 and torch.all(w[:-1] >= w[1:])
    # custom difficulties : arrays or callables
    calls = []
    def difficulty_fn(epoch, step):
        calls.append(epoch)
        return [epoch % 2, 1 - epoch % 2]
    c = Curriculum(difficulty_fn, start=0.5, length=100)
    assert c.weights().tolist() == [1., 0.]
    c.set_epoch(1)
    assert c.weights().tolist() == [0., 1.] and calls == [0, 1]
    assert _rank_quantiles(torch.tensor([1., float("nan"), 3.])).tolist() == [0., 0.5, 1.]


@pytest.mark.parametrize("mode", ["online", "preprocessed"])
def test_rarity_curriculum(paths, mode):
    dataset = load(paths, mode)
    labels = dataset.item_labels("instrument")
    curriculum = Curriculum(rarity(labels), start=0.5, length=4)
    sampler = dataset.get_sampler(curriculum=curriculum, seed=0, max_samples=2000)
    seen = []
    for epoch in range(5):
        sampler.set_epoch(epoch)
        seen.append(set(frequencies(labels, sampler).keys()))
    # common classes first, rare classes at the end
    assert seen[0] == {"piano"} and seen[-1] == set(COUNTS.keys())


def test_step_curriculum():
    curriculum = Curriculum(torch.arange(100.), start=0.1, end=1., length=4, unit="step")
    sampler = WeightedItemSampler(100, curriculum=curriculum, seed=0, num_samples=400, block_size=100)
    maxima = []
    for i, index in enumerate(sampler):
        if i % 100 == 99:
            # steps are given by the training loop (e.g. CurriculumCallback) ; blocks use the current step
            curriculum.step()
        maxima.append(index)
    blocks = [max(maxima[b * 100:(b + 1) * 100]) for b in range(4)]
    assert blocks[0] < 10 and blocks == sorted(blocks) and blocks[-1] > 70


# loss tracker

def test_loss_tracker_strategies():
    for strategy, expected in [("last", 3.), ("mean", 2.), ("max", 3.), ("ema", 0.5 * (0.5 * 1 + 0.5 * 2) + 0.5 * 3)]:
        tracker = LossTracker(4, strategy=strategy, momentum=0.5)
        for loss in [1., 2., 3.]:
            tracker.update([0], [loss])
        assert tracker.values[0].item() == pytest.approx(expected), strategy
    tracker = LossTracker(4, groups=["a", "a", "b", "b"])
    assert torch.isnan(tracker.difficulty()).all()
    tracker.update(torch.tensor([0, 2]), torch.tensor([1., 5.]))
    difficulty = tracker.difficulty()
    # unseen items get the mean of their group
    assert difficulty.tolist() == [1., 1., 5., 5.]
    stats = tracker.statistics()
    assert stats['n_seen'] == 2 and stats['groups'] == {"a": 1., "b": 5.}
    copy = LossTracker(4)
    copy.load_state_dict(tracker.state_dict())
    assert torch.equal(copy.values, tracker.values)


def test_loss_curriculum():
    tracker = LossTracker(10)
    tracker.update(torch.arange(10), torch.arange(9., -1., -1.))
    curriculum = Curriculum(tracker, start=0.2, length=10)
    # items with the lowest losses are sampled first
    assert torch.where(curriculum.weights() > 0)[0].tolist() == [8, 9]


# batches and indices

def test_balanced_batches():
    labels = ["a"] * 10 + ["b"] * 3 + ["c"] * 1
    sampler = BalancedBatchSampler(labels, classes_per_batch=2, items_per_class=3, num_batches=20, seed=0)
    batches = list(sampler)
    assert len(batches) == 20 and all(len(b) == 6 for b in batches)
    for b in batches:
        classes = [labels[i] for i in b]
        assert len(set(classes)) == 2 and all(classes.count(c) == 3 for c in set(classes))
    assert list(BalancedBatchSampler(labels, 2, 3, num_batches=20, seed=0)) == batches
    shards = [list(BalancedBatchSampler(labels, 2, 3, num_batches=10, seed=0, rank=r, world_size=2)) for r in range(2)]
    assert shards[0] + shards[1] != batches and sorted(map(tuple, shards[0] + shards[1])) == sorted(map(tuple, batches))
    with pytest.raises(ValueError):
        BalancedBatchSampler(labels, classes_per_batch=4, items_per_class=1)


@pytest.mark.parametrize("mode", ["online", "preprocessed"])
def test_index_in_patterns(paths, mode):
    dataset = load(paths, mode, output_pattern="{waveform,index}")
    assert int(dataset[3]["index"]) == 3 and int(dataset[-1]["index"]) == len(dataset) - 1
    batch = next(iter(dataset.get_loader(batch_size=4, balance_by="instrument", balanced_batches=(3, 1), seed=0)))
    labels = dataset.item_labels("instrument")
    assert sorted(labels[i] for i in batch["index"].tolist()) == sorted(COUNTS.keys())
    combined = CombinedAudioDataset([load(paths, "preprocessed", output_pattern="{waveform,index}"), load(paths, mode, output_pattern="{waveform,index}")])
    assert int(combined[len(combined) - 1]["index"]) == len(combined) - 1
    # batch features keep indices
    batch = next(iter(dataset.get_loader(batch_size=2, shuffle=False, batch_features=[Loudness(sr=SR, name="l")])))
    assert batch["index"].tolist() == [0, 1]


# lightning

def test_lightning(paths):
    pl = pytest.importorskip("pytorch_lightning")
    from acids_dataset.lightning import CurriculumCallback
    dataset = load(paths, "online", output_pattern="{waveform,index}")
    tracker = LossTracker(len(dataset), groups=dataset.item_labels("instrument"))
    curriculum = Curriculum(tracker, start=0.5, length=2)
    step_curriculum = Curriculum(torch.arange(float(len(dataset))), start=0.5, length=4, unit="step")
    loader = dataset.get_loader(batch_size=4, curriculum=curriculum, seed=0)

    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(1, 1)
        def training_step(self, batch, batch_idx):
            x = batch["waveform"].abs().mean(-1)
            losses = (self.layer(x) - 1).pow(2).reshape(-1)
            return {"loss": losses.mean(), "item_losses": losses.detach()}
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=1e-3)

    callback = CurriculumCallback(curriculum, tracker)
    step_callback = CurriculumCallback(step_curriculum, log=False, name="steps")
    trainer = pl.Trainer(max_epochs=3, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, callbacks=[callback, step_callback])
    trainer.fit(Model(), loader)
    # Lightning gives epochs to the sampler ; the callback records losses and steps
    assert loader.sampler.epoch == 2 and curriculum.current_epoch == 2
    assert int(tracker.seen.sum()) > 0 and torch.isfinite(tracker.difficulty()).all()
    assert step_curriculum.current_step == trainer.global_step > 0
    state = callback.state_dict()
    restored = CurriculumCallback(Curriculum(LossTracker(len(dataset)), length=2))
    restored.load_state_dict(state)
    assert torch.equal(restored.tracker.values, tracker.values) and restored.curriculum.current_epoch == 2
