"""Tests for the common file access interface (files, load_audio, get_fragment, get_loader) of datasets."""
import shutil
import pytest
import torch

from .datasets import get_dataset
from acids_dataset import preprocess_dataset
from acids_dataset.datasets import AudioDataset, OnlineAudioDataset, PreprocessedAudioDataset
from acids_dataset.datasets.base import AudioDataset as BaseAudioDataset
from acids_dataset.features import Loudness

SR = 44100


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("file_access")
    src = get_dataset("simple_midi")
    raw = tmp_path / src.name
    shutil.copytree(src, raw, ignore=shutil.ignore_patterns("__pycache__", "*.py", ".ad"))
    out = {}
    for name, kwargs in [("overlap", dict(chunk_length=SR, hop_length=SR // 2)), ("discard", dict(chunk_length=SR)), ("nowaveform", dict(chunk_length=SR, waveform=False))]:
        preprocess_dataset(raw, out=tmp_path / name, sample_rate=SR, channels=2, features=[Loudness(sr=SR)], force=True, **kwargs)
        out[name] = AudioDataset(tmp_path / name)
    out["online"] = AudioDataset(raw, sr=SR, channels=2, chunk_length=SR, features=[Loudness(sr=SR)], write_index=False)
    return out


def test_abstract_methods():
    for method in ["files", "load_audio", "get_fragment", "split", "load_partition", "__len__", "chunk_length"]:
        assert method in BaseAudioDataset.__abstractmethods__
    assert len(OnlineAudioDataset.__abstractmethods__) == 0 and len(PreprocessedAudioDataset.__abstractmethods__) == 0
    class Incomplete(BaseAudioDataset):
        def __len__(self): return 0
    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


@pytest.mark.parametrize("mode", ["overlap", "discard", "online"])
def test_files(datasets, mode):
    files = datasets[mode].files
    assert len(files) == 10
    for f in files:
        assert {"audio_path", "duration", "sr", "channels", "n_frames"} <= set(f.keys())
        assert f["sr"] == SR and f["channels"] == 2 and f["n_frames"] == pytest.approx(f["duration"] * SR, abs=1)
    if mode != "online":
        assert all(f["n_chunks"] == len(f["keys"]) > 0 for f in files)


@pytest.mark.parametrize("mode", ["overlap", "discard"])
def test_preprocessed_load_audio(datasets, mode):
    pre, online = datasets[mode], datasets["online"]
    online_map = {f['audio_path']: i for i, f in enumerate(online.files)}
    for file_idx in [0, 5]:
        file = pre.files[file_idx]
        online_idx = online_map[file['audio_path']]
        # whole file : stored parts match the original audio (int16 precision), missing parts are zeros
        full, reference = pre.load_audio(file_idx), online.load_audio(online_idx)
        assert full.shape == reference.shape == (2, file['n_frames'])
        stored = full.abs().sum(0) > 0
        assert stored.float().mean() > 0.9
        assert torch.allclose(full[:, stored], reference[:, stored], atol=1e-3)
        # segments, in seconds or samples
        segment = pre.load_audio(file_idx, start=1.5, length=2.)
        assert torch.equal(segment, pre.load_audio(file_idx, start=1.5, length=2 * SR))
        assert torch.allclose(segment, online.load_audio(online_idx, start=1.5, length=2.), atol=1e-3)
        # beyond the end of the file
        end = pre.load_audio(file_idx, start=file['duration'] - 0.5, length=1.)
        assert end.shape == (2, SR) and torch.all(end[:, SR // 2 + 1:] == 0)
        assert pre.load_audio(file_idx, start=file['duration'] + 1., length=100).shape == (2, 100)
        replicated = pre.load_audio(file_idx, start=file['duration'] - 0.5, length=1., pad_mode="replicate")
        n_valid = file['n_frames'] - int(round((file['duration'] - 0.5) * SR))
        assert torch.all(replicated[:, n_valid:] == replicated[:, n_valid - 1:n_valid])


def test_preprocessed_without_waveform(datasets):
    with pytest.raises(ValueError, match="without waveform"):
        datasets["nowaveform"].load_audio(0)
    assert len(datasets["nowaveform"].files) == 10


@pytest.mark.parametrize("mode", ["overlap", "online"])
def test_get_fragment(datasets, mode):
    ds = datasets[mode]
    fragment = ds.get_fragment(2, start=1., length=2.)
    assert fragment.get_metadata()['audio_path'] == ds.files[2]['audio_path']
    assert float(fragment.get_metadata()['start_pos']) == 1.
    assert fragment.get_audio("waveform").shape == (2, 2 * SR)
    assert fragment.get_buffer("waveform").sampling_rate == SR
    assert fragment.get_array("loudness") is not None
    # features can be overridden
    assert not ds.get_fragment(2, start=1., length=2., features=[]).has_buffer("loudness")
    # features computed on loaded audio match between modes
    if mode == "overlap":
        online = datasets["online"]
        online_idx = {f['audio_path']: i for i, f in enumerate(online.files)}[ds.files[2]['audio_path']]
        online_fragment = online.get_fragment(online_idx, start=1., length=2.)
        assert torch.allclose(torch.as_tensor(fragment.get_array("loudness")), torch.as_tensor(online_fragment.get_array("loudness")), atol=0.1)


@pytest.mark.parametrize("mode", ["overlap", "online"])
def test_get_loader(datasets, mode):
    loader = datasets[mode].get_loader(batch_size=4, drop_last=True)
    assert next(iter(loader)).shape == (4, 2, SR)
    loader = datasets[mode].get_loader(batch_size=2, shuffle=False)
    assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)
