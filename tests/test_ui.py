"""Tests for the dataset browser (acids_dataset.ui) : backend on pre-processed and online datasets, and gradio app."""
import os
import shutil
import pytest
import numpy as np
import torch
import torchaudio

from acids_dataset import preprocess_dataset
from acids_dataset.datasets import AudioDataset
from acids_dataset.features import Loudness
from acids_dataset.ui import DatasetBrowser

SR = 22050
META_REGEXP = ["**/{{instrument}}_*.wav"]
INSTRUMENTS = {"piano": 3, "guitar": 2, "drums": 2}


@pytest.fixture(scope="module")
def browsers(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("ui")
    raw = tmp_path / "instruments"
    os.makedirs(raw)
    for k, (inst, n) in enumerate(INSTRUMENTS.items()):
        for i in range(n):
            g = torch.Generator().manual_seed(10 * k + i)
            torchaudio.save(str(raw / f"{inst}_{i}.wav"), 0.2 * torch.rand(2, (2 + i) * SR, generator=g) - 0.1, sample_rate=SR)
    preprocess_dataset(raw, out=tmp_path / "preprocessed", chunk_length=SR // 2, sample_rate=SR, channels=1,
                       meta_regexp=META_REGEXP, features=[Loudness(sr=SR)], force=True)
    AudioDataset(tmp_path / "preprocessed").split({"train": 0.7, "valid": 0.3}, features=["instrument"], seed=0, write="by_instrument")
    online = DatasetBrowser.from_path(raw, features=["loudness"], meta_regexp=META_REGEXP, chunk_length=0.5, partition_dir=tmp_path / "partitions")
    # random partition of files : files are not shared, items of a same instrument are
    online.dataset.split({"train": 0.7, "valid": 0.3}, seed=0, write="random")
    return {"preprocessed": DatasetBrowser.from_path(tmp_path / "preprocessed"), "online": online}

MODES = ["preprocessed", "online"]


@pytest.mark.parametrize("mode", MODES)
def test_summary(browsers, mode):
    browser = browsers[mode]
    summary = browser.summary()
    assert summary['mode'] == ("online" if mode == "online" else "pre-processed")
    assert summary['n_items'] == len(browser.dataset) > 0 and summary['n_files'] == sum(INSTRUMENTS.values())
    assert summary['sr'] == SR and summary['chunk_length'] == SR // 2
    assert "loudness" in summary['features'] and summary['categorical_features'] == ["instrument"]
    assert len(summary['partitions']) == 1
    rows = browser.file_table()
    assert len(rows) == summary['n_files'] and rows[0][0] == 0 and rows[0][1].startswith("instruments/")


def test_from_path_defaults(tmp_path):
    # online datasets get a default chunk length, and the most common sampling rate of their files
    for i, sr in enumerate([22050, 22050, 44100]):
        torchaudio.save(str(tmp_path / f"{i}.wav"), torch.zeros(1, sr), sample_rate=sr)
    browser = DatasetBrowser.from_path(tmp_path)
    assert browser.sr == 22050 and browser.dataset.chunk_length == 22050
    assert not (tmp_path / ".ad").exists()


@pytest.mark.parametrize("mode", MODES)
def test_items_and_files(browsers, mode):
    browser = browsers[mode]
    item = browser.item(2)
    assert item['index'] == 2 and item['audio'].shape == (1, SR // 2) and item['sr'] == SR
    assert "loudness" in item['features'] and item['metadata']['audio_path'].startswith("instruments/")
    assert browser.item(len(browser.dataset) + 1)['index'] == 1
    sr, audio = browser.file_audio(0, start=0.5, length=1.)
    assert sr == SR and audio.shape[-1] == SR
    assert browser.file_audio(0)[1].shape[-1] == browser.dataset.files[0]['n_frames']


@pytest.mark.parametrize("mode", MODES)
def test_statistics(browsers, mode):
    browser = browsers[mode]
    assert "loudness" in browser.array_feature_names and browser.categorical_feature_names == ["instrument"]
    result = browser.feature_stats("loudness", n_items=5)
    assert result['stats']['n_items'] == 5 and result['stats']['min'] <= result['stats']['mean'] <= result['stats']['max']
    assert len(result['values']) > 0 and np.all(np.isfinite(result['values']))
    whole = browser.feature_stats("loudness", n_items=None)
    assert whole['stats']['n_items'] == len(browser.dataset)
    # statistics are reproducible
    assert browser.feature_stats("loudness", n_items=5)['stats'] == result['stats']
    counts = browser.categorical_counts("instrument")
    assert set(counts.keys()) == set(INSTRUMENTS.keys()) and sum(counts.values()) == len(browser.dataset)


@pytest.mark.parametrize("mode", MODES)
def test_augmentations(browsers, mode):
    browser = browsers[mode]
    transforms = browser.available_transforms()
    assert "gain" in transforms and "reverse" in transforms
    audio = browser.item(0)['audio']
    reversed_audio, errors = browser.augment(audio, ["reverse"], force=True)
    assert errors == [] and np.allclose(reversed_audio, audio[..., ::-1], atol=1e-4)
    augmented, errors = browser.augment(audio, ["gain", "reverse", "lambda"], force=True)
    assert augmented.shape == audio.shape
    assert len(errors) == 1 and errors[0].startswith("lambda")


def test_forced_audiomentations():
    # audiomentations transforms handle probabilities themselves : forcing them must always apply them
    from acids_dataset import transforms
    t = transforms.Reverse(p=0.)
    x = np.linspace(-1, 1, 100, dtype=np.float32)[None, None]
    assert np.allclose(t(x), x)
    assert np.allclose(t(x, _force_transform=True), x[..., ::-1])
    assert t._obj.p == 0.


@pytest.mark.parametrize("mode", MODES)
def test_partitions(browsers, mode):
    browser = browsers[mode]
    name = browser.list_partitions()[0]
    report = browser.partition_report(name)
    assert set(report['sizes'].keys()) == {"train", "valid"}
    assert sum(s['n_items'] for s in report['sizes'].values()) == len(browser.dataset)
    assert report['shared_files'] == {}
    distribution = report['distributions']['instrument']
    assert sum(sum(counts.values()) for counts in distribution.values()) == len(browser.dataset)
    if mode == "preprocessed":
        # partitions by instrument : each instrument in a single partition
        for inst in INSTRUMENTS:
            assert sum(distribution[p].get(inst, 0) > 0 for p in distribution) == 1


def test_shared_files_detection(browsers):
    # random partitions of pre-processed items share files between partitions
    browser = browsers["preprocessed"]
    browser.dataset.split({"train": 0.5, "valid": 0.5}, seed=0, write="random_items")
    report = browser.partition_report("random_items")
    assert len(report['shared_files']) == 1 and len(report['shared_files']["train / valid"]) > 0


# gradio app

@pytest.mark.parametrize("mode", MODES)
def test_app(browsers, mode):
    pytest.importorskip("gradio")
    gradio_client = pytest.importorskip("gradio_client")
    from acids_dataset.ui import build_app
    from acids_dataset.ui.app import to_gradio_audio
    sr, audio = to_gradio_audio(SR, np.ones((2, 100)) * 2.)
    assert sr == SR and audio.shape == (100, 2) and audio.dtype == np.int16 and audio.max() == 32767

    browser = browsers[mode]
    app = build_app(browser)
    app.launch(prevent_thread_lock=True, inbrowser=False, quiet=True)
    try:
        client = gradio_client.Client(app.local_url, verbose=False)
        partition = browser.list_partitions()[0]
        calls = {
            "/show_item": (1,), "/show_file": (0, 0.5, 1.), "/compute_stats": ("loudness", "100 items"),
            "/show_counts": ("instrument",), "/apply_augmentations": (0, ["gain", "reverse"], True), "/show_partition": (partition,),
        }
        for api_name, args in calls.items():
            out = client.predict(*args, api_name=api_name)
            assert out is not None
        stats, _ = client.predict("loudness", "whole dataset", api_name="/compute_stats")
        assert stats['n_items'] == len(browser.dataset)
        _, _, errors, _ = client.predict(0, ["lambda"], True, api_name="/apply_augmentations")
        assert "lambda" in errors
    finally:
        app.close()


# partition creation

def test_parse_ratios():
    assert DatasetBrowser.parse_ratios("train: 0.8, valid=0.2") == {"train": 0.8, "valid": 0.2}
    assert DatasetBrowser.parse_ratios("a:0.5; b:0.25") == {"a": 0.5, "b": 0.25}
    assert DatasetBrowser.parse_ratios({"train": 1}) == {"train": 1.}
    for bad in ["", "train 0.8", "train: 0.8, valid: 0.5", "train: -0.1", "train: 0.5, train: 0.2", ": 0.5", "train: x"]:
        with pytest.raises(ValueError):
            DatasetBrowser.parse_ratios(bad)


@pytest.mark.parametrize("mode", MODES)
def test_create_partition(browsers, mode):
    browser = browsers[mode]
    assert browser.split_features == ["original_path", "instrument"]
    existing = browser.list_partitions()
    # preview : not written
    report = browser.create_partition("train: 0.6, valid: 0.2, test: 0.2", features=["original_path"], seed=0)
    assert set(report['sizes'].keys()) == {"train", "valid", "test"} and report['shared_files'] == {}
    assert browser.list_partitions() == existing
    # reproducible with seeds
    assert browser.create_partition("train: 0.6, valid: 0.4", seed=3)['sizes'] == browser.create_partition("train: 0.6, valid: 0.4", seed=3)['sizes']
    # written partitions
    report = browser.create_partition("train: 0.5, valid: 0.5", features=["instrument"], balance_cardinality=True, tolerance=0.5, name="created")
    assert "created" in browser.list_partitions()
    assert browser.partition_report("created")['sizes'] == report['sizes']
    for inst in INSTRUMENTS:
        assert sum(report['distributions']['instrument'][p].get(inst, 0) > 0 for p in report['sizes']) == 1
    with pytest.raises(FileExistsError):
        browser.create_partition("train: 0.5", name="created")
    browser.create_partition("train: 0.5, valid: 0.5", name="created", overwrite=True)
    for kwargs in [{"features": ["loudness"]}, {"name": "../outside"}, {"name": "  "}]:
        with pytest.raises(ValueError):
            browser.create_partition("train: 0.5", **kwargs)


@pytest.mark.parametrize("mode", MODES)
def test_app_create_partition(browsers, mode):
    pytest.importorskip("gradio")
    gradio_client = pytest.importorskip("gradio_client")
    from acids_dataset.ui import build_app
    browser = browsers[mode]
    app = build_app(browser)
    app.launch(prevent_thread_lock=True, inbrowser=False, quiet=True)
    try:
        client = gradio_client.Client(app.local_url, verbose=False)
        args = ["train: 0.7, valid: 0.3", ["original_path"], 0, False, 0.1, False, "from_app", False]
        status, _, sizes, warning, _ = client.predict(*args, api_name="/preview_partition")
        assert "Preview" in status and set(sizes.keys()) == {"train", "valid"} and "from_app" not in browser.list_partitions()
        status, _, sizes, _, _ = client.predict(*args, api_name="/create_partition")
        assert "written" in status and "from_app" in browser.list_partitions()
        status, _, _, _, _ = client.predict(*args, api_name="/create_partition")
        assert "Error" in status and "already exists" in status
        status, _, _, _, _ = client.predict("train: 2", [], 0, False, 0.1, False, "", False, api_name="/preview_partition")
        assert "Error" in status
    finally:
        app.close()
