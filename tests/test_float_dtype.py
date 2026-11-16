"""float_dtype option : floating point outputs of every dataset type are cast to a given dtype."""
import numpy as np
import pytest
import torch
import torchaudio

from acids_dataset import preprocess_dataset
from acids_dataset.datasets import AudioDataset, CombinedAudioDataset, OnlineAudioDataset, PreprocessedAudioDataset
from acids_dataset.datasets.utils import parse_float_dtype
from acids_dataset.features import Loudness
from acids_dataset.transforms import Gain

SR = 22050
PATTERN = "{waveform,loudness}"


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    root = tmp_path_factory.mktemp("float_dtype")
    for folder, n_files in [("a", 2), ("b", 2)]:
        (root / folder).mkdir(exist_ok=True)
        for i in range(n_files):
            torchaudio.save(str(root / folder / f"{folder}_{i}.wav"), 0.1 * torch.randn(1, 2 * SR), sample_rate=SR)
    preprocess_dataset(root / "a", out=root / "a_pre", chunk_length=SR, sample_rate=SR, channels=1, features=[Loudness(sr=SR)], force=True)
    return {"raw": root / "b", "pre": root / "a_pre"}


def _datasets(paths, **kwargs):
    return {
        "pre": AudioDataset(paths["pre"], output_pattern=PATTERN, **kwargs),
        "online": AudioDataset(paths["raw"], sr=SR, chunk_length=SR, features=[Loudness(sr=SR)], output_pattern=PATTERN,
                               write_index=False, **kwargs),
        "combined": AudioDataset([paths["pre"], paths["raw"]], sr=SR, chunk_length=SR, features=[Loudness(sr=SR)],
                                 output_pattern=PATTERN, write_index=False, **kwargs),
    }


def test_parse_float_dtype():
    assert parse_float_dtype(None) is None
    assert parse_float_dtype("float") == torch.float32 and parse_float_dtype("torch.float16") == torch.float16
    assert parse_float_dtype(torch.float64) == torch.float64 and parse_float_dtype(np.float64) == torch.float64
    with pytest.raises(ValueError):
        parse_float_dtype("int32")
    with pytest.raises(ValueError):
        parse_float_dtype(torch.int64)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16, "bfloat16"])
def test_items(paths, dtype):
    expected = parse_float_dtype(dtype)
    for name, ds in _datasets(paths, float_dtype=dtype).items():
        assert ds.float_dtype == expected, name
        item = ds[0]
        assert item['waveform'].dtype == expected and item['loudness'].dtype == expected, name
        batch = ds[0:2]
        assert batch['waveform'].dtype == expected, name


def test_default_and_setter(paths):
    ds = _datasets(paths)["pre"]
    assert ds.float_dtype == torch.float32 and ds[0]['waveform'].dtype == torch.float32
    ds.float_dtype = "double"
    assert ds[0]['waveform'].dtype == torch.float64
    ds.float_dtype = None
    assert ds.float_dtype is None


def test_online_with_transforms(paths):
    ds = AudioDataset(paths["raw"], sr=SR, chunk_length=SR, features=[Loudness(sr=SR)], output_pattern=PATTERN,
                      transforms={'waveform': Gain()}, write_index=False, float_dtype=torch.float64)
    item = ds[0]
    assert item['waveform'].dtype == torch.float64 and item['loudness'].dtype == torch.float64


def test_loaders(paths):
    for name, ds in _datasets(paths, float_dtype=torch.float64).items():
        batch = next(iter(ds.get_loader(batch_size=2)))
        assert batch['waveform'].dtype == torch.float64, name
        # features computed on batches
        ds.output_pattern = "{waveform,bl}"
        batch = next(iter(ds.get_loader(batch_size=2, batch_features=[Loudness(sr=SR, name="bl")])))
        assert batch['waveform'].dtype == torch.float64 and batch['bl'].dtype == torch.float64, name
