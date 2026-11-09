"""AudioDataset.sample_features : loading of given features on random items."""
import pytest
import torch
import torchaudio

from acids_dataset import preprocess_dataset
from acids_dataset.datasets import AudioDataset
from acids_dataset.features import Loudness, Mel

SR = 22050


class Fail():
    """feature computation that must not be called."""
    def __call__(self, *args, **kwargs):
        raise AssertionError("feature should not be computed")


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    root = tmp_path_factory.mktemp("sample_features")
    g = torch.Generator().manual_seed(0)
    (root / "raw").mkdir(exist_ok=True)
    for i in range(6):
        torchaudio.save(str(root / "raw" / f"{i}.wav"), 0.2 * torch.randn(1, 2 * SR, generator=g), sample_rate=SR)
    preprocess_dataset(root / "raw", out=root / "pre", chunk_length=SR // 2, sample_rate=SR, channels=1, features=[Loudness(sr=SR)], force=True)
    return {"raw": root / "raw", "pre": root / "pre", "root": root}


def datasets(paths):
    online = AudioDataset(paths["raw"], sr=SR, channels=1, chunk_length=SR // 2, batch_by="dataset", write_index=False,
                          features=[Loudness(sr=SR), Mel(sr=SR, n_fft=1024)], transforms=lambda x: x * 0)
    return {"online": online, "pre": AudioDataset(paths["pre"])}


@pytest.mark.parametrize("kind", ["online", "pre"])
def test_sample_features(paths, kind):
    dataset = datasets(paths)[kind]
    batches = dataset.sample_features("loudness", max=2, batch_size=3)
    assert len(batches) == 2 and all(torch.is_tensor(b) and b.shape[0] == 3 for b in batches)
    batches = dataset.sample_features("waveform", "loudness", max=100, batch_size=4)
    # every item is loaded once
    assert sum(b["waveform"].shape[0] for b in batches) == len(dataset)
    assert set(batches[0].keys()) == {"waveform", "loudness"}
    # transforms are not applied
    assert batches[0]["waveform"].abs().sum() > 0


def test_only_requested_features(paths):
    dataset = datasets(paths)["online"]
    assert dataset._features[1].feature_name == "mel_128"
    dataset._features[1].extract = Fail()
    dataset.sample_features("loudness", max=1, batch_size=2)


@pytest.mark.parametrize("kind", ["online", "pre"])
def test_max_size_and_reduction(paths, kind):
    dataset = datasets(paths)[kind]
    item_size = (SR // 2) * 4
    batches = dataset.sample_features("waveform", max=100, batch_size=1, max_size=int(2.5 * item_size))
    assert len(batches) == 2
    assert AudioDataset._parse_size("1G") == 1024 ** 3 and AudioDataset._parse_size("500MB") == 500 * 1024 ** 2
    mean = dataset.sample_features("loudness", max=100, batch_size=4, reduction=lambda b: torch.cat(b).mean())
    assert mean.ndim == 0


def test_cache(paths):
    dataset = datasets(paths)["pre"]
    cache = paths["root"] / "cache" / "loudness.pt"
    first = dataset.sample_features("loudness", max=2, batch_size=2, cache=cache)
    assert cache.exists()
    second = dataset.sample_features("loudness", max=2, batch_size=2, cache=cache)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    # other parameters : loaded again
    assert len(dataset.sample_features("loudness", max=1, batch_size=2, cache=cache)) == 1


def test_is_preprocessed(paths):
    import acids_dataset as ad
    assert ad.is_preprocessed(paths["pre"]) and ad.is_preprocessed(str(paths["pre"]))
    assert not ad.is_preprocessed(paths["raw"])
    assert not ad.is_preprocessed(paths["root"] / "missing")
    assert not ad.is_preprocessed(paths["raw"] / "0.wav")


@pytest.mark.parametrize("kind", ["online", "pre"])
def test_sample_feature_objects(paths, kind):
    from acids_dataset.features import Peak
    from acids_dataset.transforms import NormRange
    dataset = datasets(paths)[kind]
    batches = dataset.sample_features("loudness", Peak(sr=SR), NormRange(max_value=0.5), max=2, batch_size=3)
    assert set(batches[0].keys()) == {"loudness", "peak", "normrange"}
    for b in batches:
        assert b["peak"].shape == (3,)
        assert torch.allclose(b["normrange"].abs().amax(dim=(-2, -1)), torch.full((3,), 0.5))
    # waveform and objects are consistent
    batches = dataset.sample_features("waveform", Peak(sr=SR), max=1, batch_size=4)
    assert torch.allclose(batches[0]["peak"], batches[0]["waveform"].abs().amax(dim=(-2, -1)))
    # single object : batches are its outputs ; plain callables and duplicate names
    batches = dataset.sample_features(Peak(sr=SR), max=1, batch_size=4)
    assert torch.is_tensor(batches[0]) and batches[0].shape == (4,)
    batches = dataset.sample_features(Peak(sr=SR), Peak(db=True, sr=SR), max=1, batch_size=2)
    assert set(batches[0].keys()) == {"peak", "peak_1"}


def test_sample_feature_objects_cache(paths):
    from acids_dataset.features import Peak
    dataset = datasets(paths)["pre"]
    cache = paths["root"] / "cache" / "peak.pt"
    first = dataset.sample_features(Peak(sr=SR), max=2, batch_size=2, cache=cache)
    assert all(torch.equal(a, b) for a, b in zip(first, dataset.sample_features(Peak(sr=SR), max=2, batch_size=2, cache=cache)))
    # other parameters of the feature : loaded again
    db = dataset.sample_features(Peak(db=True, sr=SR), max=2, batch_size=2, cache=cache)
    assert all(b.max() < 0 for b in db)


def test_sample_features_whole_files(paths):
    from acids_dataset.features import Peak
    dataset = AudioDataset(paths["raw"], sr=SR, channels=1, write_index=False)
    batches = dataset.sample_features("waveform", Peak(sr=SR), max=1, batch_size=2)
    assert isinstance(batches[0]["waveform"], (list, torch.Tensor)) and len(batches[0]["peak"]) == 2
