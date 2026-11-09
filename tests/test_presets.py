"""Tests for presets : resolution, values and overrides, and consistency between pre-processing and online datasets."""
import os
import shutil
import logging
import pickle
import pytest
import gin
import torch
import yaml
from pathlib import Path

from .datasets import get_dataset
from acids_dataset import preprocess_dataset, presets
from acids_dataset.datasets import AudioDataset, OnlineAudioDataset, PreprocessedAudioDataset
from acids_dataset.features import Loudness

PACKAGE_PRESETS = ["default", "rave", "after", "ddsp"]

LOUDNESS_PRESET = """
include 'default.gin'

SAMPLE_RATE = 22050
CHANNELS = 1
CHUNK_LENGTH = 22050
HOP_LENGTH = 22050

Loudness:
    sr = %SAMPLE_RATE

features.parse_features:
    features = [@features.Loudness]

OnlineAudioDataset:
    batch_by = "files"
"""


@pytest.fixture
def dataset_path(tmp_path):
    src = get_dataset("simple_midi")
    dst = tmp_path / src.name
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.py", ".ad"))
    return dst

@pytest.fixture
def local_presets(tmp_path, monkeypatch):
    """runs the test from a directory with an ad_configs folder containing custom presets."""
    cwd = tmp_path / "cwd"
    os.makedirs(cwd / "ad_configs")
    (cwd / "ad_configs" / "loudness22k.gin").write_text(LOUDNESS_PRESET)
    # local presets take precedence over package ones
    (cwd / "ad_configs" / "ddsp.gin").write_text("include 'rave.gin'\nSAMPLE_RATE = 48000\n")
    monkeypatch.chdir(cwd)
    return cwd / "ad_configs"


# resolution & values

def test_list_presets(local_presets):
    available = presets.list_presets()
    assert set(PACKAGE_PRESETS + ["loudness22k"]) <= set(available.keys())
    assert available["ddsp"] == local_presets / "ddsp.gin"
    assert available["rave"].parent == presets.ACIDS_DATASET_CONFIG_PATH
    # names, file names and paths
    assert presets.get_preset_path("rave") == presets.get_preset_path("rave.gin") == available["rave"]
    assert presets.get_preset_path(local_presets / "loudness22k.gin") == local_presets / "loudness22k.gin"
    with pytest.raises(FileNotFoundError):
        presets.get_preset_path("unknown")


def test_preset_values(local_presets):
    assert presets.get_preset_values("rave") == {"SAMPLE_RATE": 44100, "CHANNELS": 1, "CHUNK_LENGTH": 262144, "HOP_LENGTH": 131072, "DEVICE": "cpu"}
    assert presets.get_preset_values("default")["CHUNK_LENGTH"] is None
    # included presets can be overridden
    assert presets.get_preset_values("ddsp")["SAMPLE_RATE"] == 48000
    assert presets.get_preset_values("ddsp")["CHUNK_LENGTH"] == 262144
    assert presets.get_preset_values("rave", overrides=["CHUNK_LENGTH = 1024"])["CHUNK_LENGTH"] == 1024
    for name in presets.list_presets():
        presets.get_preset_values(name)


def test_preset_env_restores_config():
    gin.clear_config()
    with presets.preset_env("rave", values={"SAMPLE_RATE": 22050}):
        assert presets.query_preset_value("SAMPLE_RATE") == 22050
        assert presets.query_preset_value("CHUNK_LENGTH") == 262144
    assert presets.query_preset_value("SAMPLE_RATE") is None
    assert gin.config_str() == ""


# pre-processing

@pytest.mark.parametrize("preset,kwargs,expected", [
    ("rave", {}, {"sr": 44100, "chunk_length": 262144, "hop_length": 131072, "channels": 1}),
    ("rave", {"chunk_length": 44100, "sample_rate": 22050}, {"sr": 22050, "chunk_length": 44100, "hop_length": 22050, "channels": 1}),
    ("after", {"chunk_length": 88200}, {"sr": 44100, "chunk_length": 88200, "hop_length": 44100, "channels": 1}),
    ("default", {"chunk_length": 88200, "channels": 2}, {"sr": 44100, "chunk_length": 88200, "hop_length": 44100, "channels": 2}),
])
def test_preprocess_with_preset(dataset_path, tmp_path, preset, kwargs, expected):
    out = tmp_path / "preprocessed"
    preprocess_dataset(dataset_path, out=out, preset=preset, force=True, **kwargs)
    metadata = yaml.safe_load(open(out / "metadata.yaml"))
    assert metadata['preset'] == preset
    assert metadata['sr'] == expected['sr'] and metadata['channels'] == expected['channels']
    assert round(metadata['chunk_length'] * metadata['sr']) == expected['chunk_length']
    assert round(metadata['hop_length'] * metadata['sr']) == expected['hop_length']


def test_preprocess_preset_compatibility(dataset_path, tmp_path):
    # presets can still be given with the config keyword, with or without .gin
    for config in ["rave", "rave.gin"]:
        preprocess_dataset(dataset_path, out=tmp_path / "preprocessed", config=config, force=True)
        assert yaml.safe_load(open(tmp_path / "preprocessed" / "metadata.yaml"))['chunk_length'] * 44100 == pytest.approx(262144)
    with pytest.raises(ValueError):
        # default preset does not define a chunk length
        preprocess_dataset(dataset_path, out=tmp_path / "preprocessed", force=True)


# online datasets

def test_online_with_preset(dataset_path):
    ds = AudioDataset(dataset_path, preset="rave", write_index=False)
    assert type(ds) is OnlineAudioDataset
    assert ds.sr == 44100 and ds._n_channels == 1 and ds.chunk_length == 262144 and ds.pad_mode.name == "ZERO"
    assert ds[0].shape == (1, 262144)
    # explicit arguments override preset values, and preset values used by features
    ds = AudioDataset(dataset_path, preset="rave", sr=22050, chunk_length=1., features=["loudness"], write_index=False)
    assert ds.sr == 22050 and ds.chunk_length == 22050
    assert [type(f) for f in ds.features] == [Loudness] and ds.features[0].sr == 22050
    # additional bindings
    ds = AudioDataset(dataset_path, preset="rave", preset_overrides=["SAMPLE_RATE = 16000", "OnlineAudioDataset.batch_by = 'files'"], write_index=False)
    assert ds.sr == 16000 and ds.batch_by == "files"
    # bindings of presets do not remain in the global config
    with pytest.raises(ValueError):
        gin.query_parameter("%CHUNK_LENGTH")
    ds = AudioDataset(dataset_path, write_index=False)
    assert ds.sr is not None and ds.chunk_length is None
    # datasets with presets can be pickled for data loaders
    assert len(pickle.loads(pickle.dumps(AudioDataset(dataset_path, preset="rave", write_index=False)))) > 0


def test_online_preset_features(dataset_path, local_presets):
    ds = AudioDataset(dataset_path, preset="loudness22k", features=["mel"], write_index=False)
    # preset features are completed with given features
    assert [type(f).__name__ for f in ds.features] == ["Loudness", "Mel"]
    assert ds.features[0].sr == 22050
    assert ds.batch_by == "files" and ds.chunk_length == 22050
    feature_names = [f.feature_name for f in ds.features]
    item = ds.get(0, output_pattern="{%s}"%(",".join(["waveform"] + feature_names)))
    assert item['waveform'].shape == (1, 22050) and set(item.keys()) == {"waveform", *feature_names}
    # feature config names without preset
    ds = AudioDataset(dataset_path, features=["loudness"], sr=16000, write_index=False)
    assert ds.features[0].sr == 16000


def test_preset_consistency(dataset_path, tmp_path, local_presets):
    """a preset gives the same chunks and features in pre-processed and online modes."""
    out = tmp_path / "preprocessed"
    preprocess_dataset(dataset_path, out=out, preset="loudness22k", force=True)
    preprocessed = AudioDataset(out, preset="loudness22k", output_pattern="{waveform,loudness}")
    online = AudioDataset(dataset_path, preset="loudness22k", output_pattern="{waveform,loudness}", write_index=False)
    assert type(preprocessed) is PreprocessedAudioDataset
    assert preprocessed.chunk_length == online.chunk_length == 22050
    assert set(preprocessed.features.keys()) == {f.feature_name for f in online.features}
    online_files = {f['audio_path']: i for i, f in enumerate(online.files)}
    for index in range(0, len(preprocessed), max(1, len(preprocessed) // 5)):
        fragment = preprocessed.loader[index]
        metadata = fragment.get_metadata()
        online_fragment = online.get_fragment(online_files[metadata['audio_path']], start=float(metadata['start_pos']), length=online.chunk_length)
        assert torch.allclose(torch.as_tensor(fragment.get_audio("waveform")), torch.as_tensor(online_fragment.get_audio("waveform")), atol=1e-3)
        assert torch.allclose(torch.as_tensor(fragment.get_array("loudness")), torch.as_tensor(online_fragment.get_array("loudness")), atol=0.5)


def test_preprocessed_preset_check(dataset_path, tmp_path, caplog):
    out = tmp_path / "preprocessed"
    preprocess_dataset(dataset_path, out=out, preset="rave", chunk_length=44100, force=True)
    with caplog.at_level(logging.WARNING):
        AudioDataset(out, preset="rave", preset_overrides=["CHUNK_LENGTH = 44100", "HOP_LENGTH = 22050"])
    assert "pre-processed with" not in caplog.text
    with caplog.at_level(logging.WARNING):
        AudioDataset(out, preset="rave")
    assert "chunk_length=44100 samples, but preset rave has CHUNK_LENGTH=262144" in caplog.text
