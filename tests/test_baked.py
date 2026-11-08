"""Baked features of online datasets."""
import math
import time
import numpy as np
import pytest
import torch, torchaudio
from acids_dataset.datasets import OnlineAudioDataset
from acids_dataset.datasets.baked import read_baked_file, read_baked_metadata, slice_frames
from acids_dataset.features import Loudness, Mel, SpectralCentroid, STFT, Tempo

SR = 22050
CHUNK = 8192
PATTERN = "{waveform,spectral_centroid,mel_64,loudness}"


@pytest.fixture
def audio_dir(tmp_path):
    root = tmp_path / "data"
    (root / "sub").mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    # different sampling rates and lengths (the last file is shorter than a chunk)
    for i, (sr, duration) in enumerate([(22050, 3.), (44100, 2.5), (22050, 0.3)]):
        t = torch.arange(int(sr * duration)) / sr
        x = 0.3 * torch.sin(2 * math.pi * (200 + 300 * t) * t) * (1 + torch.sin(2 * math.pi * 2 * t)) + 0.01 * torch.randn(len(t), generator=g)
        torchaudio.save(str(root / ("sub" if i else "") / f"f{i}.wav"), x[None], sr)
    return root


def features():
    return [SpectralCentroid(sr=SR, frame_length=1024, hop_length=256), Mel(sr=SR, n_fft=1024, hop_length=256, n_mels=64),
            Loudness(sr=SR, frame_length=1024)]


def dataset(root, feats=None, **kwargs):
    return OnlineAudioDataset(root, sr=SR, chunk_length=CHUNK, batch_by="dataset", features=features() if feats is None else feats,
                              output_pattern=PATTERN, **kwargs)


def test_bake_files(audio_dir):
    dataset(audio_dir).bake_features()
    baked = sorted(str(p.relative_to(audio_dir)) for p in (audio_dir / "ad-features").rglob("*.ad"))
    assert baked == ["ad-features/f0.wav.ad", "ad-features/sub/f1.wav.ad", "ad-features/sub/f2.wav.ad"]
    metadata, arrays = read_baked_file(audio_dir / "ad-features/sub/f1.wav.ad")
    assert metadata['original_sr'] == 44100 and metadata['sr'] == SR and metadata['channels'] == 1
    assert metadata['n_samples'] == int(2.5 * SR)
    info = metadata['features']
    assert info['spectral_centroid']['feature_sr'] == SR / 256 and info['mel_64']['feature_sr'] == SR / 256
    assert info['loudness']['feature_sr'] == SR / 1024
    assert set(arrays) == {"spectral_centroid", "mel_64", "loudness"} and arrays['mel_64'].shape[:2] == (1, 64)


@pytest.mark.parametrize("file_idx, start", [(0, 0.), (0, 40 * 1024 / SR), (1, 7 * 1024 / SR), (2, 0.)])
def test_baked_equals_dynamic(audio_dir, file_idx, start):
    dynamic = dataset(audio_dir).bake_features()
    baked = dataset(audio_dir).use_baked_features()
    assert baked.baked_features == ["spectral_centroid", "mel_64", "loudness"]
    a, b = dynamic.get_fragment(file_idx, start=start, length=CHUNK), baked.get_fragment(file_idx, start=start, length=CHUNK)
    for name in ["spectral_centroid", "mel_64", "loudness"]:
        va, vb = np.asarray(a.get(name)), np.asarray(b.get(name))
        assert va.shape == vb.shape, name
        if file_idx == 2:
            continue    # file shorter than a chunk : padded frames differ
        # frames aligned with the chunk start are identical, but on the edges of chunks (padded when computed on chunks)
        inner = (slice(None),) * (va.ndim - 1) + (slice(3, -3),)
        assert np.allclose(va[inner], vb[inner], rtol=1e-4, atol=1e-4), name


def test_baked_loading(audio_dir):
    dataset(audio_dir).bake_features()
    baked = dataset(audio_dir).use_baked_features()
    item = baked[3]
    assert item['mel_64'].shape == (1, 64, 33) and item['loudness'].shape == (1, 8)
    batch = next(iter(baked.get_loader(batch_size=2, shuffle=True)))
    assert batch['spectral_centroid'].shape == (2, 29)


def test_incremental_baking(audio_dir):
    dataset(audio_dir).bake_features("mel_64")
    assert list(read_baked_metadata(audio_dir / "ad-features/f0.wav.ad")['features']) == ["mel_64"]
    stft = STFT(sr=SR, frame_length=512, mode=["db", "phase"])
    dataset(audio_dir, features() + [stft]).bake_features("loudness", "stft_db_phase")
    assert list(read_baked_metadata(audio_dir / "ad-features/f0.wav.ad")['features']) == ["mel_64", "loudness", "stft_db_phase"]
    # baked features that are not dataset features are added
    baked = dataset(audio_dir, []).use_baked_features("stft_db_phase")
    assert baked.get_fragment(0, 0.5, CHUNK).get("stft_db_phase").shape == (2, 257, 61)


def test_custom_path(audio_dir, tmp_path):
    destination = tmp_path / "baked"
    dataset(audio_dir).bake_features("mel_64", path=destination)
    assert (destination / "data/sub/f1.wav.ad").exists() and not (audio_dir / "ad-features").exists()
    assert dataset(audio_dir).use_baked_features("mel_64", path=destination).baked_features == ["mel_64"]
    with pytest.raises(ValueError):
        dataset(audio_dir).use_baked_features("mel_64")


def test_bake_errors(audio_dir):
    with pytest.raises(ValueError):
        dataset(audio_dir, [Tempo(sr=SR)]).bake_features()           # no time axis
    with pytest.raises(ValueError):
        dataset(audio_dir).bake_features("unknown")


def test_use_checks(audio_dir):
    dataset(audio_dir).bake_features()
    # different feature configuration
    with pytest.raises(ValueError):
        dataset(audio_dir, [SpectralCentroid(sr=SR, frame_length=2048)]).use_baked_features("spectral_centroid")
    # different sampling rate
    with pytest.raises(ValueError):
        OnlineAudioDataset(audio_dir, sr=16000, features=[]).use_baked_features("mel_64")
    # modified audio file
    wav, sr = torchaudio.load(str(audio_dir / "f0.wav"))
    time.sleep(0.01)
    torchaudio.save(str(audio_dir / "f0.wav"), wav * 0.5, sr)
    with pytest.raises(ValueError):
        dataset(audio_dir).use_baked_features()
    dataset(audio_dir).use_baked_features(check=False)
    dataset(audio_dir).bake_features()
    dataset(audio_dir).use_baked_features()
    # missing files raise when loading if check=False
    (audio_dir / "ad-features/sub/f1.wav.ad").unlink()
    with pytest.raises(FileNotFoundError):
        dataset(audio_dir).use_baked_features(check=False).get_fragment(1, 0., CHUNK)


def test_slice_frames():
    array = np.arange(100.)[None]
    # 10 frames per second at sr = 100 (hop of 10 samples), without frame offset
    out = slice_frames(array, start=2., length=0.5, sr=100, feature_sr=10., frame_offset=0)
    assert np.array_equal(out, np.arange(20., 25.)[None])
    # frames exceeding the file repeat the last frame
    out = slice_frames(array, start=9.8, length=0.5, sr=100, feature_sr=10., frame_offset=1)
    assert np.array_equal(out, np.array([[98., 99., 99., 99., 99., 99.]]))
