"""Tests for seeded partitions (e.g. for distributed training), and partitions of combined datasets."""
import shutil
import multiprocessing as mp
import pytest
import torch

from .datasets import get_dataset
from acids_dataset import preprocess_dataset
from acids_dataset.datasets import AudioDataset, CombinedAudioDataset

TARGET = {"train": 0.6, "valid": 0.2, "test": 0.2}
SPLIT_ARGS = {
    "random": {},
    "path": {"features": ["original_path"]},
    "balanced": {"features": ["original_path"], "balance_cardinality": True, "tolerance": 0.5},
}


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("partitions")
    src = get_dataset("simple_midi")
    raw = tmp_path / src.name
    shutil.copytree(src, raw, ignore=shutil.ignore_patterns("__pycache__", "*.py", ".ad"))
    preprocess_dataset(raw, out=tmp_path / "preprocessed", chunk_length=44100, sample_rate=44100, channels=1, force=True)
    return {"online": raw, "preprocessed": tmp_path / "preprocessed"}


def load(path):
    return AudioDataset(path, chunk_length=44100, write_index=False) if "preprocessed" not in str(path) else AudioDataset(path)


def partition_signature(partitions):
    """partitions as comparable lists, in item order."""
    signature = {}
    for name, p in partitions.items():
        if hasattr(p, "file_indices"):
            signature[name] = list(p.file_indices)
        else:
            signature[name] = [k if isinstance(k, (str, bytes)) else int(k) for k in p._subindices]
    return signature


def split_in_process(args):
    path, split_args, seed = args
    return partition_signature(load(path).split(TARGET, seed=seed, **split_args))


@pytest.mark.parametrize("mode", ["online", "preprocessed"])
@pytest.mark.parametrize("split", list(SPLIT_ARGS.keys()))
def test_seeded_splits(paths, mode, split):
    ds = load(paths[mode])
    a = partition_signature(ds.split(TARGET, seed=0, **SPLIT_ARGS[split]))
    b = partition_signature(ds.split(TARGET, seed=0, **SPLIT_ARGS[split]))
    c = partition_signature(ds.split(TARGET, seed=1, **SPLIT_ARGS[split]))
    assert a == b
    if split != "balanced":
        # balanced partitions assign largest groups first : seeds only change ties
        assert a != c
    # without seed, the global random state is used
    import random
    random.seed(3); d = partition_signature(ds.split(TARGET, **SPLIT_ARGS[split]))
    random.seed(3); e = partition_signature(ds.split(TARGET, **SPLIT_ARGS[split]))
    assert d == e


@pytest.mark.parametrize("mode", ["online", "preprocessed"])
@pytest.mark.parametrize("split", ["random", "path"])
def test_seeded_splits_across_processes(paths, mode, split):
    # spawned processes have different hash seeds (and global random states) : seeded partitions must still be identical
    with mp.get_context("spawn").Pool(3) as pool:
        signatures = pool.map(split_in_process, [(paths[mode], SPLIT_ARGS[split], 42)] * 3)
    assert all(s == signatures[0] for s in signatures)


@pytest.mark.parametrize("modes", [("preprocessed", "preprocessed"), ("online", "online"), ("preprocessed", "online")])
def test_combined_partitions(paths, modes):
    combined = CombinedAudioDataset([load(paths[m]) for m in modes], weights=[0.5, 0.5])
    partitions = combined.split({"train": 0.8, "valid": 0.2}, seed=0)
    assert set(partitions.keys()) == {"train", "valid"}
    assert len(partitions["train"]) + len(partitions["valid"]) <= len(combined)
    for p in partitions.values():
        assert p.parent is combined
        assert p[0].shape == (1, 44100) and p[-1].shape == (1, 44100)
        assert p[0:3].shape == (3, 1, 44100)
        loader = torch.utils.data.DataLoader(p, batch_size=4, sampler=p.get_sampler())
        assert next(iter(loader)).shape == (4, 1, 44100)
    # partitions of combined datasets are partitions of each dataset
    for i, d in enumerate(combined._datasets):
        assert len(partitions["train"]._datasets[i]) == len(d.split({"train": 0.8, "valid": 0.2}, seed=0)["train"])
    # feature-based partitions, written and loaded
    partitions = combined.split({"train": 0.8, "valid": 0.2}, features=["original_path"], seed=0, write="combined_split")
    loaded = combined.load_partition("combined_split")
    assert {k: len(v) for k, v in loaded.items()} == {k: len(v) for k, v in partitions.items()}
    assert loaded["valid"][0].shape == (1, 44100)


# continuous features : stratified partitions

import numpy as np
import torchaudio
from acids_dataset.features import Loudness

GAIN_SR = 22050


@pytest.fixture(scope="module")
def gain_paths(tmp_path_factory):
    """40 sines with gains uniformly spread from -40 to -5 dB."""
    root = tmp_path_factory.mktemp("gains")
    raw = root / "gains"
    raw.mkdir()
    t = torch.arange(GAIN_SR) / GAIN_SR
    for i, gain in enumerate(np.linspace(-40, -5, 40)):
        torchaudio.save(str(raw / f"file_{i:02d}.wav"), (10 ** (gain / 20) * torch.sin(2 * np.pi * 440 * t))[None], sample_rate=GAIN_SR)
    preprocess_dataset(raw, out=root / "preprocessed", chunk_length=GAIN_SR // 2, sample_rate=GAIN_SR, channels=1, 
                       features=[Loudness(sr=GAIN_SR, keep_channels=False)], force=True)
    return {"online": raw, "preprocessed": root / "preprocessed"}


def load_gains(paths, mode):
    if mode == "online":
        return AudioDataset(paths["online"], sr=GAIN_SR, chunk_length=GAIN_SR // 2, features=[Loudness(sr=GAIN_SR, keep_channels=False)], write_index=False)
    return AudioDataset(paths["preprocessed"])


def partition_loudness(partition):
    return np.array([float(partition.get(i, output_pattern="loudness")) for i in range(len(partition))])


@pytest.mark.parametrize("mode", ["online", "preprocessed"])
@pytest.mark.parametrize("features", [["loudness"], ["original_path", "loudness"]])
def test_stratified_partitions(gain_paths, mode, features):
    dataset = load_gains(gain_paths, mode)
    target = {"train": 0.5, "valid": 0.25, "test": 0.25}
    partitions = dataset.split(target, features=features, seed=0, n_bins=5)
    assert sum(len(p) for p in partitions.values()) == len(dataset)
    overall = partition_loudness(dataset)
    for name, p in partitions.items():
        assert abs(len(p) / len(dataset) - target[name]) < 0.06
        values = partition_loudness(p)
        # each partition covers the whole range of values, with a similar distribution
        assert abs(np.median(values) - np.median(overall)) < 4.
        assert values.min() < np.percentile(overall, 20) and values.max() > np.percentile(overall, 80)
    # a random split of the same sizes is much less balanced in the worst case ; here, check reproducibility
    again = dataset.split(target, features=features, seed=0, n_bins=5)
    assert partition_signature(again) == partition_signature(partitions)
    if mode == "preprocessed" and "original_path" in features:
        files = {name: {p for p, keys in dataset.feature_hash["original_path"].items() 
                        if set(map(dataset._key_to_str, keys)) & set(map(dataset._key_to_str, part._subindices))} for name, part in partitions.items()}
        assert files["train"].isdisjoint(files["valid"]) and files["train"].isdisjoint(files["test"])


@pytest.mark.parametrize("mode", ["online", "preprocessed"])
def test_stratified_partitions_errors(gain_paths, mode):
    dataset = load_gains(gain_paths, mode)
    with pytest.raises(ValueError):
        dataset.split({"train": 0.5, "valid": 0.5}, features=["loudness", "loudness_2"])
    with pytest.raises(ValueError):
        dataset.split({"train": 0.5, "valid": 0.5}, features=["unknown"])
    # partial ratios : the remaining part is left out
    partitions = dataset.split({"train": 0.5, "valid": 0.2}, features=["loudness"], seed=1)
    assert 0.6 * len(dataset) <= sum(len(p) for p in partitions.values()) <= 0.8 * len(dataset)


def test_assign_stratified():
    from acids_dataset.datasets.base import AudioDataset as Base
    import random
    groups = list(range(100))
    values = list(range(100))
    assignment = Base._assign_stratified(groups, [1.] * 100, values, {"a": 0.8, "b": 0.2}, n_bins=10, rng=random.Random(0))
    assert len(assignment["a"]) == 80 and len(assignment["b"]) == 20
    # b gets values spread over the whole range
    assert np.histogram(assignment["b"], bins=5, range=(0, 100))[0].min() >= 3
