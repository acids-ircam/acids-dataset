"""get_channels_from_path : channels of pre-processed folders, raw folders and audio files."""
import time
import torch
import torchaudio

import acids_dataset as ad
from acids_dataset import preprocess_dataset
from acids_dataset.datasets import AudioDataset

SR = 16000


def write(path, channels):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), 0.1 * torch.randn(channels, SR // 2), sample_rate=SR)


def test_channels(tmp_path):
    raw = tmp_path / "raw"
    write(raw / "a.wav", 1)
    write(raw / "sub" / "b.wav", 2)
    write(raw / "sub" / "c.flac", 2)
    (raw / "notes.txt").write_text("not audio")
    assert ad.get_channels_from_path(raw) == {1, 2}
    assert ad.get_channels_from_path(raw / "a.wav") == {1}
    assert ad.get_channels_from_path(raw, exclude=["a.wav"]) == {2}
    assert ad.get_channels_from_path(raw, filters=["*.wav"]) == {1, 2}
    assert ad.get_channels_from_path(raw, valid_exts=[".flac"]) == {2}
    # index files are used for unchanged files, headers for new or modified ones
    AudioDataset(raw, channels=None)
    write(raw / "d.wav", 4)
    assert ad.get_channels_from_path(raw) == {1, 2, 4}
    # pre-processed folders : metadata, or chunk shapes if channels were kept
    preprocess_dataset(raw, out=tmp_path / "pre", chunk_length=SR // 4, sample_rate=SR, channels=2, force=True)
    assert ad.get_channels_from_path(tmp_path / "pre") == {2}
    preprocess_dataset(raw, out=tmp_path / "pre_kept", chunk_length=SR // 4, sample_rate=SR, channels=None, force=True)
    assert ad.get_channels_from_path(tmp_path / "pre_kept") == {1, 2, 4}
    assert ad.get_channels_from_path([tmp_path / "pre", raw / "a.wav"]) == {1, 2}


def test_channels_use_index(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    for i in range(20):
        write(raw / f"{i}.wav", 1 + i % 2)
    AudioDataset(raw, channels=None)
    import acids_dataset.utils as utils
    calls = []
    original = utils.audio_info
    monkeypatch.setattr(utils, "audio_info", lambda p: calls.append(p) or original(p))
    assert ad.get_channels_from_path(raw) == {1, 2}
    assert len(calls) == 0


def test_channels_file(tmp_path, monkeypatch):
    import json
    from acids_dataset.datasets import channels_file_path
    raw = tmp_path / "raw"
    write(raw / "a.wav", 1)
    write(raw / "b.wav", 2)
    assert ad.get_channels_from_path(raw) == {1, 2}
    assert not (raw / ".ad_channels").exists()            # not written by default
    assert ad.get_channels_from_path(raw, write=True) == {1, 2}
    assert channels_file_path(raw) == (raw / ".ad_channels").resolve()
    assert json.loads((raw / ".ad_channels").read_text())["channels"] == [1, 2]
    # the channels file is read instead of scanning the folder
    import acids_dataset.datasets.base as base
    monkeypatch.setattr(base, "_scan_channels", lambda *args: (_ for _ in ()).throw(AssertionError("folder scanned")))
    write(raw / "c.wav", 4)
    assert ad.get_channels_from_path(raw) == {1, 2}
    monkeypatch.undo()
    # force scans again (and overwrites with write) ; other scan options scan again
    assert ad.get_channels_from_path(raw, force=True) == {1, 2, 4}
    assert ad.get_channels_from_path(raw) == {1, 2}
    assert ad.get_channels_from_path(raw, exclude=["a.wav"]) == {2, 4}
    assert ad.get_channels_from_path(raw, force=True, write=True) == {1, 2, 4}
    assert ad.get_channels_from_path(raw) == {1, 2, 4}
    # channels files are not written in pre-processed folders, nor in single files
    preprocess_dataset(raw, out=tmp_path / "pre", chunk_length=SR // 4, sample_rate=SR, channels=2, force=True)
    assert ad.get_channels_from_path(tmp_path / "pre", write=True) == {2}
    assert not (tmp_path / "pre" / ".ad_channels").exists()
    # index_dir
    index_dir = tmp_path / "index"
    assert ad.get_channels_from_path(raw, write=True, force=True, index_dir=index_dir) == {1, 2, 4}
    assert (index_dir / "raw.ad_channels").exists()
    # the channels file is not an audio file of the dataset
    assert AudioDataset(raw, channels=None, write_index=False).files.__len__() == 3
