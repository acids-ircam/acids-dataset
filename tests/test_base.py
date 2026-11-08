"""Tests for the AudioDataset base class : class dispatch, argument checks, presets, loaders, and the common interface,
that is tested on both pre-processed and online datasets built from the same audio files."""
import os
import logging
import pytest
import numpy as np
import torch
import torchaudio
from pathlib import Path

from acids_dataset import preprocess_dataset
from acids_dataset.datasets import CombinedAudioDataset, AudioDataset, OnlineAudioDataset, PreprocessedAudioDataset, is_preprocessed_path
from acids_dataset.datasets.base import AudioDataset as BaseAudioDataset, chunk_length_to_samples
from acids_dataset.fragments import AcidsFragment
from acids_dataset.features import Loudness

SR = 22050
CHUNK = 11025
INSTRUMENTS = {"piano": 4, "guitar": 3, "drums": 2}
META_REGEXP = ["**/{{instrument}}_*.wav"]


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("base")
    raw = tmp_path / "instruments"
    os.makedirs(raw)
    for inst_idx, (inst, n) in enumerate(INSTRUMENTS.items()):
        for i in range(n):
            g = torch.Generator().manual_seed(10 * inst_idx + i)
            torchaudio.save(str(raw / f"{inst}_{i}.wav"), 0.2 * torch.rand(1, (2 + i) * SR, generator=g) - 0.1, sample_rate=SR)
    preprocess_dataset(raw, out=tmp_path / "preprocessed", chunk_length=CHUNK, sample_rate=SR, channels=1,
                       meta_regexp=META_REGEXP, features=[Loudness(sr=SR)], force=True)
    return {"raw": raw, "preprocessed": tmp_path / "preprocessed"}


def make_dataset(paths, mode, **kwargs):
    if mode == "online":
        return AudioDataset(paths["raw"], sr=SR, channels=1, chunk_length=CHUNK, batch_by="dataset", meta_regexp=META_REGEXP,
                            features=[Loudness(sr=SR)], write_index=False, **kwargs)
    return AudioDataset(paths["preprocessed"], **kwargs)

MODES = ["preprocessed", "online"]


# helpers

def test_is_preprocessed_path(paths, tmp_path):
    assert is_preprocessed_path(paths["preprocessed"])
    assert not is_preprocessed_path(paths["raw"])
    assert not is_preprocessed_path(tmp_path / "missing")
    assert not is_preprocessed_path(paths["preprocessed"] / "metadata.yaml")
    assert not is_preprocessed_path([paths["preprocessed"]])


def test_chunk_length_to_samples():
    assert chunk_length_to_samples(1024, 44100) == 1024
    assert chunk_length_to_samples(1., 44100) == 44100
    assert chunk_length_to_samples(0.5, 22050) == 11025


# class dispatch and instantiation

def test_dispatch(paths):
    assert type(AudioDataset(paths["preprocessed"])) is PreprocessedAudioDataset
    assert type(AudioDataset(paths["raw"], write_index=False)) is OnlineAudioDataset
    assert type(AudioDataset(path=paths["raw"], write_index=False)) is OnlineAudioDataset
    assert type(AudioDataset(db_path=paths["preprocessed"])) is PreprocessedAudioDataset
    # subclasses can be instantiated directly
    assert type(PreprocessedAudioDataset(paths["preprocessed"])) is PreprocessedAudioDataset
    assert type(OnlineAudioDataset(paths["raw"], write_index=False)) is OnlineAudioDataset
    with pytest.raises(ValueError):
        AudioDataset(None)
    # lists with pre-processed datasets are combined
    combined = AudioDataset([paths["raw"], paths["preprocessed"]], write_index=False, chunk_length=0.5)
    assert type(combined) is CombinedAudioDataset
    assert [type(d) for d in combined.datasets] == [OnlineAudioDataset, PreprocessedAudioDataset]


def test_abstract_interface():
    class Incomplete(BaseAudioDataset):
        def __len__(self): return 0
    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
    for cls in [OnlineAudioDataset, PreprocessedAudioDataset, CombinedAudioDataset]:
        assert len(cls.__abstractmethods__) == 0


def test_presets_dispatch(paths):
    ds = AudioDataset(paths["raw"], preset="rave", write_index=False)
    assert type(ds) is OnlineAudioDataset and ds.chunk_length == 262144
    # presets with subclasses, and overrides without preset
    ds = OnlineAudioDataset(paths["raw"], preset="rave", chunk_length=CHUNK, write_index=False)
    assert ds.chunk_length == CHUNK
    ds = AudioDataset(paths["raw"], preset_overrides=["OnlineAudioDataset.batch_by = 'files'"], chunk_length=CHUNK, write_index=False)
    assert ds.batch_by == "files"
    assert type(AudioDataset(paths["preprocessed"], preset="default")) is PreprocessedAudioDataset


@pytest.mark.parametrize("mode", MODES)
def test_check_kwargs(paths, mode, caplog):
    with pytest.raises(TypeError, match="did you mean output_pattern"):
        make_dataset(paths, mode, output_patern="waveform")
    other_arg = {"required_fields": []} if mode == "online" else {"batch_by": "files"}
    with caplog.at_level(logging.WARNING):
        make_dataset(paths, mode, **other_arg)
    assert "%s is ignored"%list(other_arg.keys())[0] in caplog.text


# common interface

@pytest.mark.parametrize("mode", MODES)
def test_items(paths, mode):
    ds = make_dataset(paths, mode)
    assert len(ds) > 0 and ds.chunk_length == CHUNK
    assert ds[0].shape == (1, CHUNK)
    assert ds[-1].shape == (1, CHUNK)
    assert ds[0:3].shape == (3, 1, CHUNK)
    with pytest.raises(IndexError):
        ds[len(ds)]
    # output patterns and transforms
    item = ds.get(1, output_pattern="{waveform->x,loudness}")
    assert set(item.keys()) == {"x", "loudness"}
    assert torch.all(ds.get(1, transforms=lambda x: x * 0) == 0)
    ds.output_pattern = "(waveform,loudness)"
    assert isinstance(ds[0], tuple) and len(ds[0]) == 2
    with pytest.raises(AssertionError):
        ds.output_pattern = 0
    ds.output_pattern = "waveform"
    ds.transforms = lambda x: x * 0
    assert torch.all(ds[0] == 0) and ds.transforms is not None


@pytest.mark.parametrize("mode", MODES)
def test_metadata_and_features(paths, mode):
    ds = make_dataset(paths, mode)
    metadata = ds.metadata
    assert isinstance(metadata, dict)
    metadata["new_key"] = 0
    assert "new_key" not in ds.metadata, "metadata must be a copy"
    feature_names = list(ds.features.keys()) if isinstance(ds.features, dict) else [f.feature_name for f in ds.features]
    assert "loudness" in feature_names
    assert set(ds.feature_hash['instrument'].keys()) == set(INSTRUMENTS.keys())
    assert len(ds.feature_hash['original_path']) == sum(INSTRUMENTS.values())


@pytest.mark.parametrize("mode", MODES)
def test_file_access(paths, mode):
    ds = make_dataset(paths, mode)
    files = ds.files
    assert len(files) == sum(INSTRUMENTS.values())
    for f in files:
        assert {"audio_path", "duration", "sr", "channels", "n_frames"} <= set(f.keys())
        assert f["audio_path"].startswith("instruments/")
    file_idx = next(i for i, f in enumerate(files) if f["audio_path"] == "instruments/piano_2.wav")
    assert files[file_idx]["duration"] == pytest.approx(4.)
    audio = ds.load_audio(file_idx, start=1., length=CHUNK)
    assert audio.shape == (1, CHUNK)
    fragment = ds.get_fragment(file_idx, start=1., length=CHUNK)
    assert isinstance(fragment, AcidsFragment)
    assert torch.allclose(torch.as_tensor(fragment.get_audio("waveform")), audio, atol=1e-6)
    assert fragment.get_array("loudness") is not None


def test_file_access_consistency(paths):
    # pre-processed and online datasets give the same audio and features for the same files
    pre, online = make_dataset(paths, "preprocessed"), make_dataset(paths, "online")
    online_idx = {f["audio_path"]: i for i, f in enumerate(online.files)}
    for file_idx, f in enumerate(pre.files):
        a = pre.load_audio(file_idx, start=0.5, length=CHUNK)
        b = online.load_audio(online_idx[f["audio_path"]], start=0.5, length=CHUNK)
        assert torch.allclose(a, b, atol=1e-3)


@pytest.mark.parametrize("mode", MODES)
def test_search(paths, mode):
    ds = make_dataset(paths, mode)
    idx = ds.index_for_features(instrument=["piano", "drums"])
    assert set(idx.keys()) == {("piano",), ("drums",)}
    with pytest.raises(IndexError):
        ds.index_for_features(instrument="violin")
    data = ds.query_from_features(3, instrument="guitar")
    assert len(data[("guitar",)]) == 3 and data[("guitar",)][0].shape == (1, CHUNK)
    data = ds.query_from_features(2, _random=False, instrument="drums")
    assert len(data[("drums",)]) == 2
    data = ds.query_from_features(..., instrument="drums")
    if mode == "online":
        # drums last 2s and 3s : 4 and 6 chunks of 0.5s
        expected = 10
    else:
        # pre-processed chunks overlap (hop_length = chunk_length / 2)
        expected = sum(f["n_chunks"] for f in ds.files if "drums" in f["audio_path"])
    assert len(data[("drums",)]) == expected
    # combinations of several features
    idx = ds.index_for_features(instrument=["piano", "drums"], original_path="instruments/piano_0.wav")
    assert len(idx[("piano", "instruments/piano_0.wav")]) > 0 and len(idx[("drums", "instruments/piano_0.wav")]) == 0


@pytest.mark.parametrize("mode", MODES)
def test_partitions(paths, mode):
    ds = make_dataset(paths, mode)
    partitions = ds.split({"train": 0.5, "valid": 0.5}, features=["instrument"], seed=0, write=True)
    assert set(partitions.keys()) == {"train", "valid"}
    for p in partitions.values():
        assert p.is_partition and p.parent is ds and len(p) > 0
        assert p[0].shape == (1, CHUNK)
    assert sum(len(p) for p in partitions.values()) == len(ds)
    # default partition name
    assert ds.has_partition("partitions")
    loaded = ds.load_partition(None)
    assert {k: len(v) for k, v in loaded.items()} == {k: len(v) for k, v in partitions.items()}
    loaded = ds.split({"train": 0.5, "valid": 0.5}, load_if_available=True)
    assert {k: len(v) for k, v in loaded.items()} == {k: len(v) for k, v in partitions.items()}
    with pytest.raises(FileNotFoundError):
        ds.load_partition("missing")
    assert not ds.has_partition("missing")


# loaders and samplers

@pytest.mark.parametrize("mode", MODES)
def test_get_loader(paths, mode):
    ds = make_dataset(paths, mode)
    loader = ds.get_loader(batch_size=4, drop_last=True, num_workers=0, pin_memory=False)
    assert isinstance(loader.sampler, torch.utils.data.RandomSampler) and loader.drop_last
    assert next(iter(loader)).shape == (4, 1, CHUNK)
    loader = ds.get_loader(batch_size=2, shuffle=False)
    assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)
    loader = ds.get_loader(batch_size=2, max_samples=6, replacement=True)
    assert loader.sampler.replacement and len(loader.sampler) == 6
    # dataset arguments configure the dataset
    loader = ds.get_loader(batch_size=2, chunk_length=CHUNK)
    with pytest.raises(TypeError):
        ds.get_loader(batch_size=2, unknown_argument=0)


def test_configure(paths):
    pre, online = make_dataset(paths, "preprocessed"), make_dataset(paths, "online")
    assert pre.configure() is pre
    assert pre.configure(chunk_length=CHUNK) is pre
    with pytest.raises(ValueError):
        pre.configure(chunk_length=2 * CHUNK)
    with pytest.raises(TypeError):
        pre.configure(batch_by="files")
    configured = online.configure(chunk_length=2 * CHUNK, batch_by="files")
    assert configured is not online and configured.chunk_length == 2 * CHUNK and online.chunk_length == CHUNK


@pytest.mark.parametrize("mode", MODES)
def test_get_sampler(paths, mode):
    ds = make_dataset(paths, mode, max_samples=5)
    assert len(ds.get_sampler()) == 5
    assert len(ds.get_sampler(max_samples=3)) == 3
    assert ds.get_sampler(replacement=True).replacement
    assert len(make_dataset(paths, mode).get_sampler()) == len(ds)


# fragments

def test_build_fragment():
    wav = 0.1 * torch.randn(2, 1000)
    loudness = Loudness(sr=SR)
    fragment = BaseAudioDataset._build_fragment(AcidsFragment, wav, SR, "folder/file.wav", 0.5, 2., [loudness],
                                                output_type="numpy", metadata={"instrument": "piano", "missing": None})
    metadata = fragment.get_metadata()
    assert metadata["audio_path"] == "folder/file.wav" and float(metadata["start_pos"]) == 0.5
    assert float(metadata["length"]) == pytest.approx(1000 / SR) and float(metadata["file_length"]) == 2.
    assert metadata["instrument"] == "piano" and "missing" not in metadata
    assert metadata["output_type"] == "numpy" and isinstance(fragment.get_audio("waveform"), np.ndarray)
    assert np.allclose(fragment.get_audio("waveform"), wav.numpy(), atol=1e-6)
    assert loudness.read(fragment) is not None
    torch_fragment = BaseAudioDataset._build_fragment(AcidsFragment, wav, SR, "file.wav", 0., 2., [], output_type="torch")
    assert torch.is_tensor(torch_fragment.get_audio("waveform")) and not torch_fragment.has_buffer("loudness")
