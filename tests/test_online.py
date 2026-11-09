"""Tests for OnlineAudioDataset : audio backends, file scanning and index, audio loading, loading parameters
(chunk_length / batch_by / pad_mode), data loaders, AudioDataset dispatch, search and partitioning."""
import os
import json
import pickle
import shutil
import pytest
import numpy as np
import torch
import torchaudio
from pathlib import Path

from .datasets import get_dataset
from acids_dataset import preprocess_dataset
from acids_dataset.datasets import AudioDataset, OnlineAudioDataset, PreprocessedAudioDataset
from acids_dataset.datasets import online as online_module
from acids_dataset.features import Loudness
from acids_dataset.utils import audio_info, load_audio_range, PadMode

SR = 44100
INSTRUMENTS = {"piano": 6, "guitar": 4, "drums": 2}
INSTRUMENT_REGEXP = ["**/{{instrument}}_*.wav"]


# helpers

def write_audio(path, n_frames, sr=SR, channels=1, seed=0):
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    torchaudio.save(str(path), 0.5 * torch.rand(channels, n_frames, generator=g) - 0.25, sample_rate=sr)
    return path

def read_full(path):
    return load_audio_range(path)[0]

def copy_test_dataset(name, target_dir):
    # index files are written in dataset folders : work on a copy
    src = get_dataset(name)
    dst = Path(target_dir) / src.name
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.py", ".ad"))
    return dst

class CountCalls():
    def __init__(self, fn):
        self.fn = fn
        self.n = 0
    def __call__(self, *args, **kwargs):
        self.n += 1
        return self.fn(*args, **kwargs)

@pytest.fixture
def count_scans(monkeypatch):
    counter = CountCalls(online_module.audio_info)
    monkeypatch.setattr(online_module, "audio_info", counter)
    return counter

@pytest.fixture
def instrument_dataset(tmp_path):
    """piano_0..5 (1..6s), guitar_0..3 (1..4s), drums_0..1 (1..2s), and noise.wav (1s, no instrument)."""
    root = tmp_path / "instruments"
    for inst_idx, (inst, n) in enumerate(INSTRUMENTS.items()):
        for i in range(n):
            write_audio(root / f"{inst}_{i}.wav", (1 + i) * SR, seed=10 * inst_idx + i)
    write_audio(root / "noise.wav", SR, seed=100)
    return root

@pytest.fixture
def durations_dataset(tmp_path):
    durations = [0.5, 1.0, 3.2, 10.]
    root = tmp_path / "durations"
    for i, d in enumerate(durations):
        write_audio(root / f"file_{i}.wav", int(d * SR), seed=i)
    return root, durations


# audio backends

@pytest.mark.parametrize("backend", ["soundfile", "torchcodec"])
def test_audio_backends(backend):
    """partial loading must match full decoding for every supported format."""
    n_tested = 0
    for root, _, files in os.walk(get_dataset("simple")):
        for f in files:
            path = os.path.join(root, f)
            try:
                info = audio_info(path)
            except Exception:
                continue
            # torchcodec does not compensate mp3 encoder delay when seeking
            if backend == "torchcodec" and path.endswith(".mp3"):
                continue
            full, sr = load_audio_range(path, backend="soundfile")
            assert full.shape == (info['channels'], info['n_frames'])
            assert sr == info['sr']
            assert info['duration'] == pytest.approx(info['n_frames'] / info['sr'])
            frame_offset, num_frames = info['n_frames'] // 2, min(sr, info['n_frames'] // 4)
            part, part_sr = load_audio_range(path, frame_offset=frame_offset, num_frames=num_frames, backend=backend)
            assert part_sr == sr
            assert part.shape == (info['channels'], num_frames)
            assert torch.allclose(part, full[:, frame_offset:frame_offset+num_frames], atol=1e-4)
            n_tested += 1
    assert n_tested > 0


def test_load_audio_range_bounds(tmp_path):
    path = write_audio(tmp_path / "x.wav", 1000, channels=2)
    assert load_audio_range(path)[0].shape == (2, 1000)
    assert load_audio_range(path, frame_offset=900, num_frames=500)[0].shape == (2, 100)
    assert load_audio_range(path, frame_offset=2000, num_frames=500)[0].shape == (2, 0)
    assert load_audio_range(path, num_frames=0)[0].shape == (2, 0)


def test_audio_info_invalid_file(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not an audio file")
    with pytest.raises(Exception):
        audio_info(bad)


# scanning & index

@pytest.mark.parametrize("custom_index_dir", [False, True])
def test_index_location(instrument_dataset, custom_index_dir, tmp_path):
    index_dir = tmp_path / "indexes" if custom_index_dir else None
    ds = OnlineAudioDataset(instrument_dataset, index_dir=index_dir)
    index_path = ds.index_path(instrument_dataset)
    if custom_index_dir:
        assert index_path == index_dir / "instruments.ad"
        assert not (instrument_dataset / ".ad").exists()
    else:
        assert index_path == instrument_dataset / ".ad"
    assert index_path.exists()


def test_index_content(instrument_dataset):
    ds = OnlineAudioDataset(instrument_dataset, meta_regexp=INSTRUMENT_REGEXP)
    with open(ds.index_path(instrument_dataset)) as f:
        index = json.load(f)
    assert index['version'] == OnlineAudioDataset.index_version
    assert index['meta_regexp'] == INSTRUMENT_REGEXP
    assert len(index['files']) == len(ds.files) == sum(INSTRUMENTS.values()) + 1
    entry = index['files']['piano_2.wav']
    for key in ['sr', 'channels', 'n_frames', 'duration', 'codec', 'backend', 'size', 'mtime', 'metadata']:
        assert key in entry
    assert entry['n_frames'] == 3 * SR and entry['sr'] == SR and entry['channels'] == 1
    assert entry['metadata'] == {'instrument': 'piano'}
    file = next(f for f in ds.files if f['path'].endswith("piano_2.wav"))
    assert file['audio_path'] == "instruments/piano_2.wav"
    assert Path(file['path']).is_absolute()


def test_index_cache(instrument_dataset, count_scans):
    n_files = sum(INSTRUMENTS.values()) + 1
    ds = OnlineAudioDataset(instrument_dataset)
    assert count_scans.n == n_files
    # unmodified files are not re-scanned
    count_scans.n = 0
    ds_reloaded = OnlineAudioDataset(instrument_dataset)
    assert count_scans.n == 0
    assert ds_reloaded.files == ds.files
    # rescan ignores index
    OnlineAudioDataset(instrument_dataset, rescan=True)
    assert count_scans.n == n_files


def test_index_updates(instrument_dataset, count_scans):
    ds = OnlineAudioDataset(instrument_dataset, meta_regexp=INSTRUMENT_REGEXP)
    modified, removed, added = instrument_dataset / "piano_5.wav", instrument_dataset / "guitar_0.wav", instrument_dataset / "drums_10.wav"
    write_audio(modified, SR // 2, seed=1)
    os.remove(removed)
    write_audio(added, 2 * SR, seed=2)
    count_scans.n = 0
    ds_updated = OnlineAudioDataset(instrument_dataset, meta_regexp=INSTRUMENT_REGEXP)
    assert count_scans.n == 2
    files = {Path(f['path']).name: f for f in ds_updated.files}
    assert files["piano_5.wav"]['n_frames'] == SR // 2
    assert "guitar_0.wav" not in files
    assert files["drums_10.wav"]['n_frames'] == 2 * SR and files["drums_10.wav"]['metadata']['instrument'] == "drums"
    with open(ds_updated.index_path(instrument_dataset)) as f:
        assert set(json.load(f)['files'].keys()) == set(files.keys())


def test_invalid_files(instrument_dataset, count_scans):
    bad = instrument_dataset / "bad.wav"
    bad.write_bytes(b"not an audio file")
    ds = OnlineAudioDataset(instrument_dataset)
    assert [Path(p).name for p in ds.invalid_files] == ["bad.wav"]
    assert "bad.wav" not in [Path(f['path']).name for f in ds.files]
    with open(ds.index_path(instrument_dataset)) as f:
        assert "bad.wav" in json.load(f)['invalid']
    # invalid files are not re-scanned...
    count_scans.n = 0
    OnlineAudioDataset(instrument_dataset)
    assert count_scans.n == 0
    # ...unless modified
    write_audio(bad, SR)
    ds = OnlineAudioDataset(instrument_dataset)
    assert count_scans.n == 1
    assert ds.invalid_files == []
    assert "bad.wav" in [Path(f['path']).name for f in ds.files]
    # test dataset with unreadable file
    ds = OnlineAudioDataset(copy_test_dataset("simple", instrument_dataset.parent))
    assert any(p.endswith("empty.wav") for p in ds.invalid_files)


def test_write_index(instrument_dataset):
    ds = OnlineAudioDataset(instrument_dataset, write_index=False)
    assert not ds.index_path(instrument_dataset).exists()
    assert len(ds.files) == sum(INSTRUMENTS.values()) + 1


@pytest.mark.parametrize("index_content", ["corrupted", "version"])
def test_invalid_index(instrument_dataset, index_content, count_scans):
    ds = OnlineAudioDataset(instrument_dataset)
    index_path = ds.index_path(instrument_dataset)
    if index_content == "corrupted":
        index_path.write_text("{not json")
    else:
        index = json.loads(index_path.read_text())
        index['version'] = -1
        index_path.write_text(json.dumps(index))
    count_scans.n = 0
    ds_reloaded = OnlineAudioDataset(instrument_dataset)
    assert count_scans.n == len(ds.files)
    assert ds_reloaded.files == ds.files
    assert json.loads(index_path.read_text())['version'] == OnlineAudioDataset.index_version


def test_scan_filters(tmp_path):
    root = tmp_path / "data"
    write_audio(root / "a" / "x.wav", SR)
    write_audio(root / "b" / "y.wav", SR)
    os.makedirs(root / "c")
    torchaudio.save(str(root / "c" / "z.flac"), torch.zeros(1, SR), sample_rate=SR)
    names = lambda ds: sorted(Path(f['path']).name for f in ds.files)
    assert names(OnlineAudioDataset(root, write_index=False)) == ["x.wav", "y.wav", "z.flac"]
    assert names(OnlineAudioDataset(root, filters=["a/*"], write_index=False)) == ["x.wav"]
    assert names(OnlineAudioDataset(root, exclude=["b/*"], write_index=False)) == ["x.wav", "z.flac"]
    assert names(OnlineAudioDataset(root, valid_exts=[".flac"], write_index=False)) == ["z.flac"]


def test_multiple_roots(tmp_path):
    write_audio(tmp_path / "r1" / "d1" / "x.wav", SR)
    write_audio(tmp_path / "r2" / "d2" / "x.wav", 2 * SR)
    ds = OnlineAudioDataset([tmp_path / "r1" / "d1", tmp_path / "r2" / "d2"], chunk_length=1., batch_by="dataset")
    assert sorted(f['audio_path'] for f in ds.files) == ["d1/x.wav", "d2/x.wav"]
    assert (tmp_path / "r1" / "d1" / ".ad").exists() and (tmp_path / "r2" / "d2" / ".ad").exists()
    assert len(ds) == 3
    # folders with the same name would share audio paths and index files
    write_audio(tmp_path / "r3" / "d1" / "x.wav", SR)
    with pytest.raises(ValueError):
        OnlineAudioDataset([tmp_path / "r1" / "d1", tmp_path / "r3" / "d1"])


def test_meta_regexp(instrument_dataset):
    ds = OnlineAudioDataset(instrument_dataset, meta_regexp=INSTRUMENT_REGEXP)
    metadata = {Path(f['path']).name: f['metadata'] for f in ds.files}
    assert metadata["piano_0.wav"] == {"instrument": "piano"}
    assert metadata["noise.wav"] == {"instrument": None}
    # changing meta_regexp updates metadata, even with an existing index
    ds = OnlineAudioDataset(instrument_dataset, meta_regexp=["**/*_{{take}}.wav"])
    metadata = {Path(f['path']).name: f['metadata'] for f in ds.files}
    assert metadata["guitar_3.wav"] == {"take": "3"}
    assert OnlineAudioDataset(instrument_dataset).files[0]['metadata'] == {}


def test_invalid_paths(tmp_path):
    with pytest.raises(FileNotFoundError):
        OnlineAudioDataset(tmp_path / "missing")
    os.makedirs(tmp_path / "empty")
    with pytest.raises(RuntimeError):
        OnlineAudioDataset(tmp_path / "empty")


# audio loading

@pytest.fixture
def loading_dataset(tmp_path):
    root = tmp_path / "loading"
    write_audio(root / "a_mono.wav", 2 * SR, sr=SR, channels=1, seed=0)
    write_audio(root / "b_stereo.wav", 96000, sr=48000, channels=2, seed=1)
    return root


@pytest.mark.parametrize("channels", [None, 1, 4])
def test_load_audio_native_sr(loading_dataset, channels):
    ds = OnlineAudioDataset(loading_dataset, channels=channels)
    for i, file in enumerate(ds.files):
        full = read_full(file['path'])
        sr = file['sr']
        # whole file
        wav = ds.load_audio(i)
        n_channels = channels or file['channels']
        assert wav.shape == (n_channels, file['n_frames'])
        # chunks : float lengths are seconds, int lengths are samples
        chunk = ds.load_audio(i, start=0.5, length=0.25)
        assert torch.equal(chunk, ds.load_audio(i, start=0.5, length=int(0.25 * sr)))
        assert chunk.shape == (n_channels, int(0.25 * sr))
        expected = full[:, int(0.5 * sr):int(0.75 * sr)]
        expected = expected[torch.arange(n_channels) % expected.shape[0]]
        assert torch.allclose(chunk, expected)


@pytest.mark.parametrize("sr", [22050, 48000])
def test_load_audio_resampling(loading_dataset, sr):
    ds = OnlineAudioDataset(loading_dataset, sr=sr, channels=None)
    for i, file in enumerate(ds.files):
        chunk = ds.load_audio(i, start=0.5, length=0.25)
        assert chunk.shape[-1] == int(0.25 * sr)
        assert ds.load_audio(i, length=1000).shape[-1] == 1000
        wav = ds.load_audio(i)
        assert abs(wav.shape[-1] - file['n_frames'] * sr / file['sr']) <= 1
        if file['sr'] == sr:
            assert torch.allclose(wav, read_full(file['path']))


@pytest.mark.timeout(60)
@pytest.mark.parametrize("pad_mode", ["zero", "repeat", "reflect", "replicate"])
def test_padding(tmp_path, pad_mode):
    root = tmp_path / "padding"
    write_audio(root / "a_short.wav", 1000)
    write_audio(root / "b_one_sample.wav", 1)
    write_audio(root / "c_empty.wav", 0)
    ds = OnlineAudioDataset(root, pad_mode=pad_mode)
    x = read_full(ds.files[0]['path'])
    wav = ds.load_audio(0, length=3000)
    assert wav.shape == (1, 3000)
    assert torch.equal(wav[:, :1000], x)
    if pad_mode == "zero":
        assert torch.all(wav[:, 1000:] == 0)
    elif pad_mode == "replicate":
        assert torch.all(wav[:, 1000:] == x[:, -1:])
    elif pad_mode == "repeat":
        assert torch.equal(wav[:, 1000:1999], x[:, :999])
    elif pad_mode == "reflect":
        assert torch.equal(wav[:, 1000:1999], x[:, :-1].flip(-1))
    # pad_mode can be overridden
    assert torch.all(ds.load_audio(0, length=3000, pad_mode="zero")[:, 1000:] == 0)
    # degenerate cases : single sample, empty file, start after the end of the file
    for file_idx, start in [(1, 0.), (2, 0.), (0, 10.)]:
        assert ds.load_audio(file_idx, start=start, length=3000).shape == (1, 3000)
    resampled = OnlineAudioDataset(root, pad_mode=pad_mode, sr=22050)
    assert resampled.load_audio(0, start=10., length=3000).shape == (1, 3000)
    with pytest.raises(ValueError):
        OnlineAudioDataset(root, pad_mode="discard")


@pytest.mark.parametrize("output_type", ["torch", "numpy"])
def test_get_fragment(instrument_dataset, output_type):
    loudness = Loudness()
    ds = OnlineAudioDataset(instrument_dataset, features=[loudness], meta_regexp=INSTRUMENT_REGEXP, sr=22050, output_type=output_type)
    file_idx = next(i for i, f in enumerate(ds.files) if f['path'].endswith("piano_3.wav"))
    fragment = ds.get_fragment(file_idx, start=1.0, length=2.0)
    metadata = fragment.get_metadata()
    assert metadata['audio_path'] == "instruments/piano_3.wav"
    assert float(metadata['start_pos']) == 1.0
    assert float(metadata['length']) == pytest.approx(2.0)
    assert float(metadata['file_length']) == pytest.approx(4.0)
    assert metadata['instrument'] == "piano"
    assert metadata['output_type'] == output_type
    assert fragment.get_buffer("waveform").sampling_rate == 22050
    audio = fragment.get_audio("waveform")
    assert isinstance(audio, torch.Tensor if output_type == "torch" else np.ndarray)
    assert audio.shape == (1, 44100)
    assert torch.allclose(torch.as_tensor(audio, dtype=torch.float32), ds.load_audio(file_idx, start=1.0, length=2.0), atol=1e-6)
    assert loudness.read(fragment) is not None
    # features can be overridden
    fragment = ds.get_fragment(file_idx, start=1.0, length=2.0, features=[])
    assert not fragment.has_buffer(loudness.feature_name)
    # files without regexp values have no metadata entry
    noise_idx = next(i for i, f in enumerate(ds.files) if f['path'].endswith("noise.wav"))
    assert "instrument" not in ds.get_fragment(noise_idx, length=0.5).get_metadata()


# loading parameters & items

def test_sampling_rate(tmp_path):
    root = tmp_path / "mixed"
    write_audio(root / "a.wav", SR, sr=44100)
    write_audio(root / "b.wav", 48000, sr=48000)
    ds = OnlineAudioDataset(root)
    assert ds.sr is None
    for chunk_length in [1., 44100]:
        with pytest.raises(ValueError):
            ds.configure(chunk_length=chunk_length)
    ds = OnlineAudioDataset(root, sr=22050, chunk_length=1., batch_by="files")
    assert ds.sr == 22050 and ds.chunk_length == 22050
    assert all(ds[i].shape == (1, 22050) for i in range(len(ds)))
    assert OnlineAudioDataset(root, filters=["a.wav"]).sr == 44100


def test_loading_params(durations_dataset):
    root, durations = durations_dataset
    ds = OnlineAudioDataset(root)
    # without chunk_length, items are whole files
    assert ds.chunk_length is None and ds.batch_by == "files"
    assert len(ds) == len(durations)
    assert [ds[i].shape[-1] for i in range(len(ds))] == [f['n_frames'] for f in ds.files]
    configured = ds.configure(chunk_length=1., batch_by="dataset", pad_mode="reflect")
    # configure returns a copy
    assert configured is not ds and ds.chunk_length is None
    assert configured.chunk_length == SR and configured.batch_by == "dataset" and configured.pad_mode == PadMode.REFLECT
    assert len(configured.configure(chunk_length=SR)) == len(configured)
    metadata = configured.metadata
    assert metadata['chunk_length'] == SR and metadata['batch_by'] == "dataset" and metadata['pad_mode'] == "reflect"
    assert metadata['n_files'] == len(durations) and metadata['n_seconds'] == pytest.approx(sum(durations), abs=1e-3)
    # constructor parameters
    ds = OnlineAudioDataset(root, chunk_length=1., batch_by="files")
    assert len(ds) == len(durations)
    # invalid parameters
    with pytest.raises(TypeError):
        ds.configure(unknown=0)
    for kwargs in [{"chunk_length": 0.}, {"chunk_length": -1}, {"batch_by": "unknown"}, {"pad_mode": "discard"}]:
        with pytest.raises(ValueError):
            ds.configure(**kwargs)


@pytest.mark.parametrize("pad_mode", ["zero", "repeat", "reflect", "replicate"])
def test_batch_by(durations_dataset, pad_mode):
    root, durations = durations_dataset
    ds = OnlineAudioDataset(root, pad_mode=pad_mode)
    # files : one item per file
    files_ds = ds.configure(chunk_length=1., batch_by="files")
    assert files_ds.batch_by == "files" and len(files_ds) == len(durations)
    # dataset : ~ total duration / chunk_length, at least one item per file
    dataset_ds = ds.configure(chunk_length=1., batch_by="dataset")
    assert len(dataset_ds) == sum(max(1, round(d)) for d in durations)
    assert abs(len(dataset_ds) - sum(durations)) <= len(durations)
    for d in [files_ds, dataset_ds]:
        for i in range(len(d)):
            file_idx, start = d.chunk_position(i)
            duration = d.files[file_idx]['duration']
            assert start >= 0
            if duration >= 1.:
                assert start + 1. <= duration + 1e-6, "chunk must not exceed file"
            else:
                assert start == 0
            assert d[i].shape == (1, SR)
    # auto : few long files -> dataset ; many short files -> files
    assert ds.configure(chunk_length=1., batch_by="auto").batch_by == "dataset"
    assert ds.configure(chunk_length=4., batch_by="auto").batch_by == "files"
    assert OnlineAudioDataset(root, auto_ratio=10.).configure(chunk_length=1., batch_by="auto").batch_by == "files"


def test_chunk_position(durations_dataset):
    root, durations = durations_dataset
    ds = OnlineAudioDataset(root, chunk_length=1., batch_by="dataset")
    long_file = len(durations) - 1
    long_items = ds.item_indices_for_files([long_file])
    assert len(long_items) == 10
    # slots cover the file, in order
    starts = [ds.chunk_position(i, random_offset=False)[1] for i in long_items]
    assert starts == sorted(starts) and starts[0] == 0. and starts[-1] == pytest.approx(9.)
    assert all(ds.chunk_position(i)[0] == long_file for i in long_items)
    # random offsets
    files_ds = ds.configure(batch_by="files")
    random_starts = set(files_ds.chunk_position(long_file)[1] for _ in range(20))
    assert len(random_starts) > 1 and all(0 <= s <= 9. for s in random_starts)
    assert files_ds.chunk_position(long_file, random_offset=False) == (long_file, 0.)
    # indexing
    assert ds.chunk_position(-1)[0] == long_file
    for index in [len(ds), -len(ds) - 1]:
        with pytest.raises(IndexError):
            ds.chunk_position(index)


@pytest.mark.parametrize("output_pattern", ["waveform", "(waveform,loudness)", "{waveform,loudness}", "{waveform->x,}"])
def test_getitem(instrument_dataset, output_pattern):
    ds = OnlineAudioDataset(instrument_dataset, output_pattern=output_pattern, features=[Loudness()], chunk_length=0.5, batch_by="files")
    item = ds[0]
    if output_pattern == "waveform":
        assert item.shape == (1, SR // 2)
    elif output_pattern.startswith("("):
        assert item[0].shape == (1, SR // 2) and item[1] is not None
    elif "->" in output_pattern:
        assert item['x'].shape == (1, SR // 2)
    else:
        assert item['waveform'].shape == (1, SR // 2) and item['loudness'] is not None
    # get can override output pattern and transforms
    assert ds.get(0, output_pattern="waveform").shape == (1, SR // 2)
    assert torch.all(ds.get(0, output_pattern="waveform", transforms=lambda x: x * 0) == 0)
    ds.transforms = lambda x: x * 0
    assert torch.all(ds.get(0, output_pattern="waveform") == 0)


# loaders

def test_get_loader(durations_dataset):
    root, durations = durations_dataset
    ds = OnlineAudioDataset(root)
    loader = ds.get_loader(batch_size=2, chunk_length=0.5, batch_by="dataset", shuffle=False)
    assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)
    assert len(loader.dataset) == sum(max(1, round(2 * d)) for d in durations)
    batches = list(loader)
    assert all(b.shape[1:] == (1, SR // 2) for b in batches)
    assert sum(b.shape[0] for b in batches) == len(loader.dataset)
    # loader parameters do not modify the dataset
    assert ds.chunk_length is None
    # sampling
    loader = ds.get_loader(batch_size=2, chunk_length=0.5, max_samples=5, drop_last=True)
    assert isinstance(loader.sampler, torch.utils.data.RandomSampler)
    assert len(loader.sampler) == 5 and len(list(loader)) == 2
    loader = ds.get_loader(batch_size=4, chunk_length=0.5, batch_by="files", max_samples=8, replacement=True)
    assert loader.sampler.replacement and sum(b.shape[0] for b in loader) == 8
    with pytest.raises(ValueError):
        ds.get_loader(batch_size=2)
    with pytest.raises(TypeError):
        ds.get_loader(batch_size=2, chunk_length=0.5, unknown=0)


@pytest.mark.parametrize("output_pattern", ["waveform", "{waveform,loudness}"])
def test_get_loader_workers(tmp_path, output_pattern):
    dataset_path = copy_test_dataset("simple_midi", tmp_path)
    ds = AudioDataset(dataset_path, output_pattern=output_pattern, features=[Loudness()], sr=22050)
    loader = ds.get_loader(batch_size=4, chunk_length=0.5, batch_by="dataset", num_workers=2, drop_last=True)
    batch = next(iter(loader))
    if output_pattern == "waveform":
        assert batch.shape == (4, 1, 11025)
    else:
        assert batch['waveform'].shape == (4, 1, 11025) and batch['loudness'].shape[0] == 4


def test_pickle(instrument_dataset):
    ds = OnlineAudioDataset(instrument_dataset, meta_regexp=INSTRUMENT_REGEXP, features=[Loudness()], chunk_length=0.5)
    ds_copy = pickle.loads(pickle.dumps(ds))
    assert type(ds_copy) is OnlineAudioDataset
    assert ds_copy.files == ds.files and len(ds_copy) == len(ds)
    assert ds_copy[0].shape == (1, SR // 2)
    partition = ds.split({"train": 0.5, "valid": 0.5})["train"]
    partition_copy = pickle.loads(pickle.dumps(partition))
    assert partition_copy.file_indices == partition.file_indices and len(partition_copy) == len(partition)


# dispatch

def test_audio_dataset_dispatch(tmp_path):
    dataset_path = copy_test_dataset("simple_midi", tmp_path)
    preprocessed_path = tmp_path / "preprocessed"
    preprocess_dataset(dataset_path, out=preprocessed_path, chunk_length=44100, sample_rate=44100, channels=1, force=True)
    assert type(AudioDataset(dataset_path)) is OnlineAudioDataset
    assert type(AudioDataset(path=dataset_path)) is OnlineAudioDataset
    assert type(AudioDataset([dataset_path])) is OnlineAudioDataset
    preprocessed = AudioDataset(preprocessed_path)
    assert type(preprocessed) is PreprocessedAudioDataset
    assert isinstance(preprocessed, AudioDataset)
    with pytest.raises(ValueError):
        AudioDataset([preprocessed_path])
    with pytest.raises(ValueError):
        AudioDataset(None)

    # preprocessed loader
    assert preprocessed.chunk_length == 44100
    loader = preprocessed.get_loader(batch_size=4, drop_last=True)
    batch = next(iter(loader))
    assert batch.shape == (4, 1, 44100)
    preprocessed.get_loader(batch_size=4, chunk_length=1.)
    with pytest.raises(ValueError):
        preprocessed.get_loader(batch_size=4, chunk_length=2.)


# search

def test_search(instrument_dataset):
    ds = AudioDataset(instrument_dataset, meta_regexp=INSTRUMENT_REGEXP, chunk_length=1., batch_by="dataset")
    assert ds.path_features == ["original_path", "instrument"]
    assert set(ds.feature_hash['instrument'].keys()) == set(INSTRUMENTS.keys())
    assert len(ds.feature_hash['original_path']) == len(ds.files)
    with pytest.raises(ValueError):
        ds.index_for_features(loudness=0)
    with pytest.raises(IndexError):
        ds.index_for_features(instrument="violin")

    idx = ds.index_for_features(instrument=["piano", "drums"])
    assert set(idx.keys()) == {("piano",), ("drums",)}
    assert len(idx[("piano",)]) == INSTRUMENTS["piano"]
    assert all(ds.files[i]['metadata']['instrument'] == "piano" for i in idx[("piano",)])
    data = ds.query_from_features(3, instrument="guitar")
    assert len(data[("guitar",)]) == 3
    # drums files last 1s and 2s
    data = ds.query_from_features(..., instrument="drums")
    assert len(data[("drums",)]) == 3
    assert all(x.shape == (1, SR) for x in data[("drums",)])
    data = ds.query_from_features(2, _random=False, instrument="drums")
    assert len(data[("drums",)]) == 2
    with pytest.raises(IndexError):
        ds.item_indices_for_files([len(ds.files)])


def test_search_multiple_features(instrument_dataset):
    ds = AudioDataset(instrument_dataset, meta_regexp=["**/{{instrument}}_{{take}}.wav"], chunk_length=1.)
    idx = ds.index_for_features(instrument=["piano", "guitar"], take=[0, 3])
    assert set(idx.keys()) == {("piano", 0), ("piano", 3), ("guitar", 0), ("guitar", 3)}
    assert all(len(v) == 1 for v in idx.values())
    assert ds.files[next(iter(idx[("guitar", 3)]))]['path'].endswith("guitar_3.wav")
    assert ds.index_for_features(instrument="drums", take=0)[("drums", 0)] == ds.index_for_features(original_path="instruments/drums_0.wav")[("instruments/drums_0.wav",)]


# partitioning

@pytest.mark.parametrize("batch_by", ["files", "dataset"])
@pytest.mark.parametrize("features,balance_cardinality", [(None, False), (["instrument"], False), (["instrument"], True), (["original_path"], False)])
def test_split(instrument_dataset, batch_by, features, balance_cardinality):
    ds = AudioDataset(instrument_dataset, meta_regexp=INSTRUMENT_REGEXP, chunk_length=1., batch_by=batch_by)
    target = {"train": 0.7, "valid": 0.3}
    partitions = ds.split(target, features=features, balance_cardinality=balance_cardinality, write="test_split")
    assert set(partitions.keys()) == set(target.keys())
    files = {k: set(v.file_indices) for k, v in partitions.items()}
    assert files["train"].isdisjoint(files["valid"])
    for p in partitions.values():
        assert p.is_partition and p.parent is ds
        assert p.batch_by == ds.batch_by
        assert len(p) == len(p.item_indices_for_files(p.file_indices))
        for i in range(len(p)):
            file_idx, _ = p.chunk_position(i)
            assert file_idx in p.file_indices
        # feature hash and search are restricted to partition files
        assert set(sum(p.feature_hash['original_path'].values(), [])) == set(p.file_indices)
        with pytest.raises(AssertionError):
            p.split(target)
    if features is None or features == ["original_path"]:
        assert files["train"] | files["valid"] == set(ds.file_indices)
    else:
        # feature values belong to a single partition ; files without values are left out
        for inst in INSTRUMENTS:
            in_partitions = [k for k, v in partitions.items() if inst in v.feature_hash['instrument']]
            assert len(in_partitions) == 1
        assert sum(len(v) for v in files.values()) == sum(INSTRUMENTS.values())
    if features is None:
        # random partitions follow target ratios on items
        assert abs(len(partitions["train"]) / len(ds) - target["train"]) < 0.25

    # write / load
    assert ds.has_partition("test_split")
    assert (instrument_dataset / ".ad_partitions" / "test_split.json").exists()
    loaded = ds.load_partition("test_split")
    assert {k: set(v.file_indices) for k, v in loaded.items()} == files
    assert ds.split(target, load_if_available="test_split").keys() == loaded.keys()
    # loading parameters of partitions can be configured
    loader = partitions["train"].get_loader(batch_size=2, chunk_length=0.5)
    assert next(iter(loader)).shape == (2, 1, SR // 2)


@pytest.mark.parametrize("balance_cardinality", [False, True])
def test_split_partial_ratios(instrument_dataset, balance_cardinality):
    ds = AudioDataset(instrument_dataset, meta_regexp=INSTRUMENT_REGEXP, chunk_length=1.)
    partitions = ds.split({"train": 0.5, "valid": 0.2}, features=["instrument"] if balance_cardinality else None, balance_cardinality=balance_cardinality)
    n_files = sum(len(p.file_indices) for p in partitions.values())
    assert 0 < n_files < len(ds.files)


def test_partition_files(instrument_dataset, tmp_path):
    # default partition folder is in index_dir if given
    index_dir = tmp_path / "indexes"
    ds = AudioDataset(instrument_dataset, index_dir=index_dir, chunk_length=1.)
    assert ds.partition_dir == index_dir / "partitions"
    ds.split({"train": 0.5, "valid": 0.5}, write=True)
    assert (index_dir / "partitions" / "partitions.json").exists()
    assert ds.has_partition("partitions") and not ds.has_partition("other")
    with pytest.raises(FileNotFoundError):
        ds.load_partition("other")
    # files are referred with their audio paths
    with open(index_dir / "partitions" / "partitions.json") as f:
        content = json.load(f)
    assert all(p.startswith("instruments/") for p in sum(content.values(), []))
    with pytest.raises(IndexError):
        ds.get_subdataset([len(ds.files)], check=True)
    assert ds.get_subdataset([2, 0, 0]).file_indices == [0, 2]


def test_split_from_meta(tmp_path):
    root = tmp_path / "data"
    for part, n in [("train", 4), ("valid", 2)]:
        for i in range(n):
            write_audio(root / part / f"file_{i}.wav", SR, seed=i)
    partition_dir = tmp_path / "partitions"
    ds = AudioDataset(root, meta_regexp=["**/{{partition}}/*.wav"], chunk_length=1., partition_dir=partition_dir)
    partitions = ds.split({"train": 0.5, "valid": 0.5}, write=True)
    assert len(partitions["train"]) == 4 and len(partitions["valid"]) == 2
    assert (partition_dir / "partitions.json").exists()
    # partition feature is ignored if use_meta_if_available is False
    partitions = ds.split({"train": 0.5, "valid": 0.5}, use_meta_if_available=False)
    assert len(partitions["train"]) == 3
    # removed files are ignored when loading partitions
    os.remove(root / "train" / "file_0.wav")
    ds = AudioDataset(root, meta_regexp=["**/{{partition}}/*.wav"], chunk_length=1., partition_dir=partition_dir)
    assert len(ds.load_partition()["train"]) == 3
    with pytest.raises(FileNotFoundError):
        ds.load_partition(check=True)


# arguments

def test_unknown_arguments(instrument_dataset, tmp_path, caplog):
    with pytest.raises(TypeError, match="output_pattern"):
        AudioDataset(instrument_dataset, output_format="(waveform,loudness)")
    with pytest.raises(TypeError, match="chunk_length"):
        AudioDataset(instrument_dataset, chunk_lenght=1.)
    # arguments of pre-processed datasets are ignored with a warning
    import logging
    with caplog.at_level(logging.WARNING):
        AudioDataset(instrument_dataset, required_fields=["waveform"], write_index=False)
    assert "required_fields is ignored for OnlineAudioDataset" in caplog.text
    preprocessed = tmp_path / "preprocessed"
    preprocess_dataset(instrument_dataset, out=preprocessed, chunk_length=SR, sample_rate=SR, force=True)
    with caplog.at_level(logging.WARNING):
        AudioDataset(preprocessed, batch_by="files", meta_regexp=INSTRUMENT_REGEXP)
    assert "batch_by is ignored for PreprocessedAudioDataset" in caplog.text
    with pytest.raises(TypeError, match="output_pattern"):
        AudioDataset(preprocessed, output_format="waveform")


def test_feature_dict(instrument_dataset):
    ds = AudioDataset(instrument_dataset, chunk_length=0.5, write_index=False, 
                      features={'loud': Loudness(), 'loud_cfg': "loudness"}, output_pattern="(waveform,loud,loud_cfg)")
    assert [f.feature_name for f in ds.features] == ["loud", "loud_cfg"]
    waveform, loud, loud_cfg = ds[0]
    assert waveform.shape == (1, SR // 2) and loud.shape == loud_cfg.shape
    # dict features with presets
    ds = AudioDataset(instrument_dataset, preset="rave", features={'l': Loudness()}, write_index=False)
    assert [f.feature_name for f in ds.features] == ["l"]


# whole files (no chunk_length)

def test_whole_files(durations_dataset):
    root, durations = durations_dataset
    ds = OnlineAudioDataset(root, write_index=False)
    assert len(ds) == len(durations) and ds.batch_by == "files"
    assert [ds[i].shape[-1] for i in range(len(ds))] == [int(d * SR) for d in durations]
    assert ds.chunk_position(1) == (1, 0.)
    # whole files cannot be batched
    with pytest.raises(ValueError):
        ds.get_loader(batch_size=2)
    loader = ds.get_loader(batch_size=1, shuffle=False)
    assert [b.shape[-1] for b in loader] == [int(d * SR) for d in durations]
    assert len(ds.get_loader(batch_size=2, shuffle=False, collate_fn=list)) == 2
    # partitions keep whole files ; chunk_length can be given later
    partitions = ds.split({"train": 0.5, "valid": 0.5}, seed=0)
    assert sum(len(p) for p in partitions.values()) == len(durations)
    assert len(ds.configure(chunk_length=1., batch_by="dataset")) == sum(max(1, round(d)) for d in durations)


def test_whole_files_mixed_sample_rates(tmp_path):
    write_audio(tmp_path / "a.wav", SR, sr=44100)
    write_audio(tmp_path / "b.wav", 48000, sr=48000)
    ds = OnlineAudioDataset(tmp_path, write_index=False)
    # without sr, files are loaded at their sampling rate
    assert len(ds) == 2 and ds[0].shape[-1] == SR and ds[1].shape[-1] == 48000


# silence

def test_loudness_threshold(tmp_path):
    # files : 1s of silence, 1s of noise, 1s of silence
    x = torch.zeros(1, 3 * SR)
    x[:, SR:2 * SR] = 0.1 * torch.randn(1, SR)
    write_audio(tmp_path / "a.wav", 3 * SR)
    torchaudio.save(str(tmp_path / "a.wav"), x, sample_rate=SR)
    plain = OnlineAudioDataset(tmp_path, chunk_length=SR // 4, batch_by="files", write_index=False)
    filtered = OnlineAudioDataset(tmp_path, chunk_length=SR // 4, batch_by="files", loudness_threshold=-50, max_retries=50, write_index=False)
    plain_silent = sum(float(plain[0].abs().max()) == 0 for _ in range(40))
    filtered_silent = sum(float(filtered[0].abs().max()) == 0 for _ in range(40))
    assert plain_silent > 10 and filtered_silent == 0
    assert filtered.metadata['loudness_threshold'] == -50
    # whole files are never redrawn ; silent files give silent chunks after max_retries
    torchaudio.save(str(tmp_path / "a.wav"), torch.zeros(1, SR), sample_rate=SR)
    silent = OnlineAudioDataset(tmp_path, chunk_length=SR // 4, loudness_threshold=-50, max_retries=3, rescan=True, write_index=False)
    assert silent[0].abs().max() == 0


def test_loudness_floor():
    from acids_dataset.features.loudness import LoudnessScript
    silence = torch.zeros(1, SR)
    assert float(Loudness(sr=SR, keep_channels=False)(silence)) == -70.
    assert float(Loudness(sr=SR, keep_channels=False, min_loudness=None)(silence)) == float("-inf")
    assert float(Loudness(sr=SR, keep_channels=False, min_loudness=-90.)(silence)) == -90.
    scripted = torch.jit.script(Loudness(sr=SR, frame_length=512).script())
    assert torch.all(scripted(silence) == -70.)
    noise = 0.1 * torch.randn(1, SR)
    assert float(Loudness(sr=SR, keep_channels=False)(noise)) > -70.
