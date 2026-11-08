"""CombinedAudioDataset as an AudioDataset : items, files, loaders, configuration, weights, partitions, presets."""
import collections
import pytest
import torch
import torchaudio

from acids_dataset import preprocess_dataset
from acids_dataset.datasets import AudioDataset, CombinedAudioDataset, OnlineAudioDataset, PreprocessedAudioDataset
from acids_dataset.features import Loudness

SR = 22050


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    root = tmp_path_factory.mktemp("combined")
    for folder, n_files in [("a", 4), ("b", 3)]:
        for i in range(n_files):
            (root / folder).mkdir(exist_ok=True)
            torchaudio.save(str(root / folder / f"{folder}_{i}.wav"), 0.1 * torch.randn(1, (2 + i) * SR), sample_rate=SR)
    preprocess_dataset(root / "a", out=root / "a_pre", chunk_length=SR, sample_rate=SR, channels=1, features=[Loudness(sr=SR)], force=True)
    return {"raw": root / "b", "pre": root / "a_pre"}


@pytest.fixture
def combined(paths):
    return AudioDataset([paths["pre"], paths["raw"]], sr=SR, chunk_length=SR, features=[Loudness(sr=SR)], write_index=False)


def test_dispatch_and_interface(combined, paths):
    assert type(combined) is CombinedAudioDataset and isinstance(combined, AudioDataset)
    assert [type(d) for d in combined.datasets] == [PreprocessedAudioDataset, OnlineAudioDataset]
    assert len(CombinedAudioDataset.__abstractmethods__) == 0
    assert len(combined) == sum(len(d) for d in combined.datasets)
    assert combined.chunk_length == SR and combined.path == [d.path for d in combined.datasets]
    assert "loudness" in combined.features and combined.metadata['n_items'] == len(combined)
    with pytest.raises(ValueError):
        CombinedAudioDataset([paths["pre"]])
    # path keyword
    assert type(AudioDataset(path=[paths["pre"], paths["raw"]], sr=SR, chunk_length=SR, write_index=False)) is CombinedAudioDataset


def test_items(combined):
    n_first = len(combined.datasets[0])
    assert combined[0].shape == (1, SR) and combined[-1].shape == (1, SR)
    assert torch.equal(combined[1], combined.datasets[0][1])
    assert combined[n_first].shape == combined.datasets[1][0].shape
    assert combined[n_first - 2:n_first + 2].shape == (4, 1, SR)
    item = combined.get(n_first, output_pattern="{waveform,loudness}")
    assert set(item.keys()) == {"waveform", "loudness"}
    with pytest.raises(IndexError):
        combined[len(combined)]
    assert torch.all(combined.get(0, transforms=lambda x: x * 0) == 0)


def test_files(combined):
    files = combined.files
    assert len(files) == sum(len(d.files) for d in combined.datasets)
    assert [f['dataset'] for f in files] == [0] * len(combined.datasets[0].files) + [1] * len(combined.datasets[1].files)
    n_first = len(combined.datasets[0].files)
    assert torch.allclose(combined.load_audio(n_first, start=0.5, length=0.25), combined.datasets[1].load_audio(0, start=0.5, length=0.25))
    assert torch.allclose(torch.as_tensor(combined.load_audio(1, length=0.5)), torch.as_tensor(combined.datasets[0].load_audio(1, length=0.5)))
    fragment = combined.get_fragment(n_first, start=0., length=0.5)
    assert fragment.get_audio("waveform").shape == (1, SR // 2)
    with pytest.raises(IndexError):
        combined.load_audio(len(files))


def test_loader_and_configure(combined):
    loader = combined.get_loader(batch_size=4, drop_last=True)
    assert next(iter(loader)).shape == (4, 1, SR)
    # loading arguments configure online datasets, and are ignored for pre-processed ones
    configured = combined.configure(chunk_length=SR, batch_by="files")
    assert configured is not combined and combined.datasets[1].batch_by == "dataset"
    assert configured.datasets[1].batch_by == "files"
    assert len(configured) == len(combined.datasets[0]) + len(combined.datasets[1].files)
    assert configured.chunk_length == SR
    # chunk lengths of pre-processed datasets are fixed : items could not be batched together
    with pytest.raises(ValueError):
        combined.configure(chunk_length=SR // 2)


def test_weights(paths):
    dataset = CombinedAudioDataset([paths["pre"], paths["raw"]], weights=[0.9, 0.1], sr=SR, chunk_length=SR, write_index=False)
    n_first = len(dataset.datasets[0])
    counts = collections.Counter(int(i >= n_first) for i in dataset.get_sampler(replacement=True, max_samples=4000, valid=True))
    assert abs(counts[0] / 4000 - 0.9) < 0.03
    # without weights, items are sampled uniformly
    uniform = CombinedAudioDataset([paths["pre"], paths["raw"]], sr=SR, chunk_length=SR, write_index=False)
    counts = collections.Counter(int(i >= n_first) for i in uniform.get_sampler(replacement=True, max_samples=4000, valid=True))
    assert abs(counts[0] / 4000 - n_first / len(uniform)) < 0.03


def test_partitions(combined):
    partitions = combined.split({"train": 0.7, "valid": 0.3}, features=["original_path"], seed=0, write="combined_split")
    assert all(type(p) is CombinedAudioDataset and p.is_partition for p in partitions.values())
    assert sum(len(p) for p in partitions.values()) == len(combined)
    loaded = combined.load_partition("combined_split")
    assert {k: len(v) for k, v in loaded.items()} == {k: len(v) for k, v in partitions.items()}
    assert next(iter(partitions["train"].get_loader(batch_size=2))).shape == (2, 1, SR)


def test_preset(paths):
    dataset = AudioDataset([paths["pre"], paths["raw"]], preset="default", preset_overrides=["CHUNK_LENGTH = %d"%SR, "SAMPLE_RATE = %d"%SR], write_index=False)
    assert type(dataset) is CombinedAudioDataset and dataset.datasets[1].chunk_length == SR and dataset.datasets[1].sr == SR


def test_devices(combined):
    assert combined.feature_devices == {"cpu"}
    assert combined.to("cpu") is combined
