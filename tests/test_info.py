"""Tests for the info command, on pre-processed datasets and raw audio folders."""
import io
import shutil
import pytest
from pathlib import Path

from .datasets import get_dataset
from acids_dataset import preprocess_dataset
from acids_dataset import info


@pytest.fixture
def dataset_path(tmp_path):
    src = get_dataset("simple")
    dst = tmp_path / src.name
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.py", ".ad"))
    return dst


class FakeStdin(io.StringIO):
    def isatty(self):
        return True


@pytest.mark.parametrize("answer", ["y", "n"])
def test_info_online_ask_index(dataset_path, monkeypatch, capsys, answer):
    answers = iter(["maybe", answer])
    monkeypatch.setattr("sys.stdin", FakeStdin())
    monkeypatch.setattr("builtins.input", lambda *args: next(answers))
    dataset = info.info_online(dataset_path)
    out = capsys.readouterr().out
    assert (dataset_path / ".ad").exists() == (answer == "y")
    assert ("not written" in out) == (answer == "n")
    assert "n_files" in out and len(dataset.files) == 11
    # existing index is updated without asking
    if answer == "y":
        monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("should not ask"))
        info.info_online(dataset_path)


@pytest.mark.parametrize("write_index", [True, False])
def test_info_online_flags(dataset_path, tmp_path, monkeypatch, capsys, write_index):
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("should not ask"))
    index_dir = tmp_path / "indexes"
    info.info_online(dataset_path, write_index=write_index, index_dir=index_dir)
    assert (index_dir / f"{dataset_path.name}.ad").exists() == write_index
    assert not (dataset_path / ".ad").exists()


def test_info_online_non_interactive(dataset_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO())
    info.info_online(dataset_path)
    assert not (dataset_path / ".ad").exists()


def test_info_online_content(dataset_path, capsys):
    dataset = info.info_online(dataset_path, write_index=False, meta_regexp=["**/chord_tutti_simple_{{idx}}.*"],
                               files=True, check_metadata=True, idx=[0, 999], key=["simple_dataset/long/chord_tutti_simple_9.mp3", "unknown.wav"])
    out = capsys.readouterr().out
    summary = info.online_summary(dataset)
    assert summary['n_files'] == 11 and summary['sample_rates'] == {44100: 10, 48000: 1}
    assert [Path(p).name for p in summary['invalid_files']] == ["empty.wav"]
    assert summary['metadata'] == {'idx': 10}
    assert "No metadata discrepencies" in out
    assert "simple_dataset/long/chord_tutti_simple_0.flac: " in out
    assert "index 999 not in dataset" in out and "key unknown.wav not in dataset" in out
    assert "codec: MPEG_LAYER_III" in out
    # files without metadata values are reported
    info.info_online(dataset_path, write_index=False, meta_regexp=["**/long/chord_tutti_simple_{{idx}}.*"], check_metadata=True)
    assert "simple_dataset/short/chord_tutti_simple_2.wav: ['idx'] missing" in capsys.readouterr().out


def test_info_preprocessed(dataset_path, tmp_path, capsys):
    out_path = tmp_path / "preprocessed"
    preprocess_dataset(dataset_path, out=out_path, preset="rave", force=True)
    info.info_preprocessed(out_path, files=True, check_metadata=True, idx=[0, 99999], key=["000000001"])
    out = capsys.readouterr().out
    assert "'preset': 'rave'" in out and "simple_dataset/short/empty.wav" in out
    assert "simple_dataset/long/chord_tutti_simple_0.flac: " in out and " chunks" in out
    assert "No metadata discrepencies" in out
    assert "--INDEX 0" in out and "index 99999 not in dataset" in out
    assert "--KEY 000000001" in out and "audio_path" in out
