"""Partitions by files or by items (split(by=...)), for online and pre-processed datasets."""
import pytest
import torch
import torchaudio

from acids_dataset import preprocess_dataset
from acids_dataset.datasets import AudioDataset
from acids_dataset.features import Loudness

SR = 8000
CHUNK = SR // 2
# few files of different lengths : partitions by files cannot follow ratios
UNEVEN = [1., 2., 5.]
# many files of the same length : partitions by files follow ratios
EVEN = [1.] * 10
RATIOS = {"train": 0.8, "valid": 0.2}


def make_dataset(root, durations):
    g = torch.Generator().manual_seed(0)
    (root / "raw").mkdir()
    for i, d in enumerate(durations):
        torchaudio.save(str(root / "raw" / f"file_{i}.wav"), 0.1 * (i + 1) * torch.randn(1, int(d * SR), generator=g), sample_rate=SR)
    preprocess_dataset(root / "raw", out=root / "pre", chunk_length=CHUNK, hop_length=CHUNK, sample_rate=SR, channels=1, force=True)
    return {"online": root / "raw", "preprocessed": root / "pre"}


@pytest.fixture(scope="module")
def uneven(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("uneven"), UNEVEN)


@pytest.fixture(scope="module")
def even(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("even"), EVEN)


def load(paths, mode, **kwargs):
    if mode == "online":
        return AudioDataset(paths["online"], sr=SR, chunk_length=CHUNK, batch_by="dataset", write_index=False, **kwargs)
    return AudioDataset(paths["preprocessed"], **kwargs)


def files_of(partition, mode):
    """set of files of the items of a partition."""
    if mode == "online":
        return {f for f, _ in partition._all_items()}
    return set(partition._file_groups().keys())


def is_split_by_files(partitions, mode):
    files = [files_of(p, mode) for p in partitions.values()]
    return all(not (a & b) for i, a in enumerate(files) for b in files[i + 1:])


@pytest.mark.parametrize("mode", ["online", "preprocessed"])
def test_split_by_files(uneven, mode):
    ds = load(uneven, mode)
    parts = ds.split(RATIOS, seed=0, by="files", use_meta_if_available=False)
    assert is_split_by_files(parts, mode)
    assert sum(len(p) for p in parts.values()) == len(ds)
    if mode == "online":
        assert all(p.item_subindices is None for p in parts.values())


@pytest.mark.parametrize("mode", ["online", "preprocessed"])
def test_split_by_items(uneven, mode):
    ds = load(uneven, mode)
    parts = ds.split(RATIOS, seed=0, by="items", use_meta_if_available=False)
    assert not is_split_by_files(parts, mode)
    for name, r in RATIOS.items():
        assert abs(len(parts[name]) - r * len(ds)) <= 1
    if mode == "online":
        items = [i for p in parts.values() for i in p.item_subindices]
        assert sorted(items) == sorted(ds._all_items())
        # items of partitions are the items of the dataset
        part = parts["valid"]
        file_idx, chunk_idx = part.item_subindices[0]
        assert part.chunk_position(0, random_offset=False) == (file_idx, ds._chunk_slot(file_idx, chunk_idx)[0])
        assert part[0].shape == (1, CHUNK)


@pytest.mark.parametrize("mode", ["online", "preprocessed"])
def test_split_auto(uneven, even, mode):
    # few uneven files : partitions are split by items
    parts = load(uneven, mode).split(RATIOS, seed=0, use_meta_if_available=False)
    assert not is_split_by_files(parts, mode)
    # many even files : partitions are split by files
    ds = load(even, mode)
    parts = ds.split(RATIOS, seed=0, use_meta_if_available=False)
    assert is_split_by_files(parts, mode)
    assert len(parts["train"]) == 0.8 * len(ds)
    with pytest.raises(ValueError):
        ds.split(RATIOS, by="chunks")


def test_split_auto_stratified(uneven, even):
    loudness = Loudness(sr=SR, keep_channels=False)
    parts = load(uneven, "online", features=[loudness]).split(RATIOS, seed=0, features=["loudness"])
    assert not is_split_by_files(parts, "online")
    parts = load(even, "online", features=[loudness]).split(RATIOS, seed=0, features=["loudness"])
    assert is_split_by_files(parts, "online")
    parts = load(uneven, "online", features=[loudness]).split(RATIOS, seed=0, features=["loudness"], by="files")
    assert is_split_by_files(parts, "online")


def test_online_item_partitions_persistence(uneven, tmp_path):
    ds = load(uneven, "online", partition_dir=tmp_path)
    parts = ds.split(RATIOS, seed=0, by="items", write="by_items")
    loaded = ds.load_partition("by_items")
    for name in RATIOS:
        assert loaded[name].item_subindices == parts[name].item_subindices
    # item partitions are only valid for their loading parameters
    with pytest.raises(ValueError):
        parts["train"].configure(chunk_length=CHUNK * 2)
    assert len(parts["train"].configure(chunk_length=CHUNK)) == len(parts["train"])
    with pytest.raises(ValueError):
        ds.configure(chunk_length=CHUNK * 2).load_partition("by_items")
