"""Tests for slice and sequence indexing of datasets."""
import shutil
import pytest
import numpy as np
import torch

from .datasets import get_dataset
from acids_dataset import preprocess_dataset
from acids_dataset.datasets import AudioDataset, CombinedAudioDataset
from acids_dataset.features import Loudness

CHUNK = 22050


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("indexing")
    src = get_dataset("simple_midi")
    raw = tmp_path / src.name
    shutil.copytree(src, raw, ignore=shutil.ignore_patterns("__pycache__", "*.py", ".ad"))
    preprocess_dataset(raw, out=tmp_path / "preprocessed", preset="default", chunk_length=CHUNK, sample_rate=22050, channels=1,
                       features=[Loudness(sr=22050)], force=True)
    preprocessed = AudioDataset(tmp_path / "preprocessed")
    online = AudioDataset(raw, sr=22050, channels=1, chunk_length=CHUNK, features=[Loudness(sr=22050)], write_index=False)
    return {"preprocessed": preprocessed, "online": online}


@pytest.mark.parametrize("mode", ["preprocessed", "online"])
def test_slices(datasets, mode):
    ds = datasets[mode]
    n = len(ds)
    assert ds[0:4].shape == (4, 1, CHUNK)
    assert ds[:3].shape == (3, 1, CHUNK)
    assert ds[-2:].shape == (2, 1, CHUNK)
    assert ds[::max(1, n // 3)].shape[0] == len(range(0, n, max(1, n // 3)))
    assert ds[5:1:-1].shape == (4, 1, CHUNK)
    assert ds[n:n + 10] == []
    # pre-processed items are deterministic : batches match items
    if mode == "preprocessed":
        assert torch.equal(ds[2:4][1], ds[3])
        assert torch.equal(ds[-1], ds[n - 1])


@pytest.mark.parametrize("mode", ["preprocessed", "online"])
@pytest.mark.parametrize("index", [[0, 2, 5], (1, 3), np.array([0, 4]), torch.tensor([2, 1])])
def test_sequences(datasets, mode, index):
    batch = datasets[mode][index]
    assert batch.shape == (len(index), 1, CHUNK)


@pytest.mark.parametrize("mode", ["preprocessed", "online"])
def test_scalar_indices(datasets, mode):
    ds = datasets[mode]
    for index in [np.int64(1), torch.tensor(1), -1]:
        assert ds[index].shape == (1, CHUNK)
    with pytest.raises(IndexError):
        ds[len(ds)]


@pytest.mark.parametrize("mode", ["preprocessed", "online"])
def test_slices_with_patterns(datasets, mode):
    ds = datasets[mode]
    batch = ds.get(slice(0, 3), output_pattern="{waveform,loudness}")
    assert batch['waveform'].shape == (3, 1, CHUNK) and batch['loudness'].shape[0] == 3
    batch = ds.get([0, 1], output_pattern="(waveform,loudness)")
    assert batch[0].shape == (2, 1, CHUNK) and batch[1].shape[0] == 2
    batch = ds.get(slice(0, 2), output_pattern="waveform", transforms=lambda x: x * 0)
    assert torch.all(batch == 0)


def test_partitions_and_combined(datasets):
    preprocessed, online = datasets["preprocessed"], datasets["online"]
    # partitions
    train = preprocessed.split({"train": 0.5, "valid": 0.5})["train"]
    assert train[:3].shape == (3, 1, CHUNK) and torch.equal(train[1:3][0], train[1])
    online_train = online.split({"train": 0.5, "valid": 0.5})["train"]
    assert online_train[-3:].shape == (3, 1, CHUNK)
    # combined datasets
    combined = CombinedAudioDataset([preprocessed, online])
    n = len(combined)
    assert combined[len(preprocessed) - 2:len(preprocessed) + 2].shape == (4, 1, CHUNK)
    assert combined[[0, n - 1]].shape == (2, 1, CHUNK)
    assert torch.equal(combined[-n], preprocessed[0])
    assert combined[-1].shape == (1, CHUNK)
    with pytest.raises(IndexError):
        combined[n]


def test_uncollatable_items(tmp_path):
    # items of different shapes are returned as lists
    import torchaudio
    for i, n in enumerate([1000, 2000]):
        torchaudio.save(str(tmp_path / f"{i}.wav"), torch.zeros(1, n), sample_rate=22050)
    ds = AudioDataset(tmp_path, chunk_length=500, batch_by="files", write_index=False)
    batch = ds.get(slice(0, 2), output_pattern="waveform")
    assert batch.shape == (2, 1, 500)
    full = [ds.load_audio(i) for i in range(2)]
    from acids_dataset.datasets.utils import collate_items
    assert isinstance(collate_items(full), list) and len(collate_items(full)) == 2
