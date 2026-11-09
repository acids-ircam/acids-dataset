"""resampling of pre-processed datasets (sr argument of PreprocessedAudioDataset)."""
import pytest
import torch
import torchaudio

from acids_dataset import preprocess_dataset
from acids_dataset.datasets import AudioDataset, PreprocessedAudioDataset, CombinedAudioDataset
from acids_dataset.features import Loudness

SR = 22050


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    root = tmp_path_factory.mktemp("pre_resampling")
    g = torch.Generator().manual_seed(0)
    (root / "raw").mkdir(exist_ok=True)
    t = torch.arange(2 * SR) / SR
    for i in range(3):
        # band-limited signal, such that resampling is accurate
        wav = 0.3 * torch.sin(2 * torch.pi * (220 * (i + 1)) * t)[None] + 0.01 * torch.randn(1, 2 * SR, generator=g)
        torchaudio.save(str(root / "raw" / f"{i}.wav"), wav, sample_rate=SR)
    preprocess_dataset(root / "raw", out=root / "pre", chunk_length=SR // 2, sample_rate=SR, channels=1, features=[Loudness(sr=SR)], force=True)
    return {"raw": root / "raw", "pre": root / "pre"}


@pytest.mark.parametrize("sr", [16000, 44100])
def test_items_are_resampled(paths, sr):
    ref = AudioDataset(paths["pre"])
    ds = AudioDataset(paths["pre"], sr=sr)
    assert isinstance(ds, PreprocessedAudioDataset)
    assert ds.sr == sr and ds.stored_sr == SR and ds.is_resampled
    assert ds.metadata['sr'] == sr and ds.metadata['stored_sr'] == SR
    assert ds.chunk_length == -(-(SR // 2) * sr // SR)
    x, x_ref = ds[0], ref[0]
    assert x.shape[-1] == ds.chunk_length
    assert torch.allclose(x, torchaudio.functional.resample(x_ref, SR, sr), atol=1e-5)
    # stored features are unchanged
    ds.output_pattern = ref.output_pattern = "loudness"
    assert torch.equal(ds[0], ref[0])
    # batches
    ds.output_pattern = "waveform"
    batch = next(iter(ds.get_loader(batch_size=4)))
    assert batch.shape[-1] == ds.chunk_length


@pytest.mark.parametrize("sr", [16000, 44100])
def test_load_audio(paths, sr):
    ref = AudioDataset(paths["pre"])
    ds = AudioDataset(paths["pre"], sr=sr)
    whole = ds.load_audio(0)
    expected = torchaudio.functional.resample(ref.load_audio(0), SR, sr)
    assert whole.shape == expected.shape
    assert torch.allclose(whole, expected, atol=1e-4)
    # ranges match the resampled whole file (no edge effects)
    part = ds.load_audio(0, start=0.5, length=1000)
    start = int(0.5 * sr)
    assert part.shape[-1] == 1000
    assert torch.allclose(part, whole[..., start:start + 1000], atol=1e-3)
    assert ds.get_fragment(0, length=0.25).get_audio("waveform").shape[-1] == int(0.25 * sr)


def test_same_sr_is_not_resampled(paths):
    ds = AudioDataset(paths["pre"], sr=SR)
    assert not ds.is_resampled and 'stored_sr' not in ds.metadata
    with pytest.raises(ValueError):
        AudioDataset(paths["pre"], sr=-1)


def test_combined(paths):
    ds = CombinedAudioDataset([paths["pre"], paths["raw"]], sr=16000, channels=1, chunk_length=8000, write_index=False)
    assert all(d.sr == 16000 for d in ds.datasets)
    assert ds.chunk_length == 8000
    assert next(iter(ds.get_loader(batch_size=4))).shape[-1] == 8000
