"""Batch features : features computed on whole batches in the main process, after transforms of the waveform. Features
computed item by item are also computed after transforms of the waveform."""
import pytest
import torch
import torchaudio

from acids_dataset import preprocess_dataset
from acids_dataset.datasets import AudioDataset, CombinedAudioDataset
from acids_dataset.datasets.batch import parse_output_pattern, BatchFeatureLoader
from acids_dataset.features import Loudness, Mel, Width, ModuleEmbedding
from acids_dataset import transforms as adt

SR = 22050


class Half():
    """deterministic, picklable transform."""
    def __call__(self, x):
        return x * 0.5


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    root = tmp_path_factory.mktemp("batch")
    g = torch.Generator().manual_seed(0)
    for i in range(6):
        (root / "raw").mkdir(exist_ok=True)
        torchaudio.save(str(root / "raw" / f"{i}.wav"), 0.2 * torch.randn(2, 2 * SR, generator=g), sample_rate=SR)
    preprocess_dataset(root / "raw", out=root / "pre", chunk_length=SR // 2, sample_rate=SR, channels=2, features=[Loudness(sr=SR)], force=True)
    return {"raw": root / "raw", "pre": root / "pre"}


def features():
    return [Loudness(sr=SR, keep_channels=False), Mel(sr=SR, n_fft=1024), Width(), ModuleEmbedding(module_path="tests/scripted/conv_embedding.ts", sr=SR, name="emb")]


def online(paths, **kwargs):
    kwargs = {"sr": SR, "channels": 1, "chunk_length": SR // 2, "batch_by": "dataset", "write_index": False, **kwargs}
    return AudioDataset(paths["raw"], **kwargs)


def test_parse_output_pattern():
    assert parse_output_pattern("waveform") == ("single", [("waveform", "waveform")])
    assert parse_output_pattern("(waveform, loudness[0])") == ("tuple", [("waveform", "waveform"), ("loudness[0]", "loudness[0]")])
    assert parse_output_pattern("{waveform->x, mel[0:2]->m, loudness}") == ("dict", [("waveform", "x"), ("mel[0:2]", "m"), ("loudness", "loudness")])


def test_batch_equals_items(paths):
    """without transforms, batch features are the features computed item by item."""
    pattern = "{waveform,loudness,mel_128,width,emb}"
    dataset = online(paths, features=features(), output_pattern=pattern)
    reference = next(iter(dataset.get_loader(batch_size=4, shuffle=False)))
    batch = next(iter(dataset.get_loader(batch_size=4, shuffle=False, batch_features=["loudness", "mel_128", "width", "emb"])))
    assert set(batch.keys()) == set(reference.keys())
    for k in reference:
        assert batch[k].shape == reference[k].shape, k
        assert torch.allclose(batch[k].float(), reference[k].float(), atol=1e-4), k


def test_features_after_transforms(paths):
    dataset = online(paths, features=[Loudness(sr=SR, keep_channels=False)], output_pattern="{waveform->x,loudness}", transforms={"x": Half()})
    batch = next(iter(dataset.get_loader(batch_size=4, shuffle=False, batch_features=["loudness"])))
    # chunks are drawn at random positions : the reference is computed on the audio of the batch, before its transform
    loudness = Loudness(sr=SR, keep_channels=False)
    for i in range(4):
        # a half amplitude is 6.02 dB quieter : batch features are computed on transformed audio
        assert abs(float(loudness(batch["x"][i] * 2) - batch["loudness"][i]) - 6.0206) < 1e-3
    expected = torch.stack([Loudness(sr=SR, keep_channels=False)(batch["x"][i]) for i in range(4)])
    assert torch.allclose(batch["loudness"], expected, atol=1e-4)


@pytest.mark.parametrize("batch_transforms", [False, True])
def test_patterns_and_transforms(paths, batch_transforms):
    dataset = online(paths, features=[Loudness(sr=SR, keep_channels=False), Mel(sr=SR, n_fft=1024)])
    # tuple
    dataset.output_pattern, dataset.transforms = "(waveform,loudness,mel_128[0])", [Half(), None, None]
    w, l, m = next(iter(dataset.get_loader(batch_size=2, shuffle=False, batch_features=["loudness", "mel_128"], batch_transforms=batch_transforms)))
    assert w.shape == (2, 1, SR // 2) and l.shape == (2,) and m.shape[:2] == (2, 128)
    plain = online(paths)
    assert torch.allclose(w, plain[0:2] * 0.5, atol=1e-6)
    # single
    dataset.output_pattern, dataset.transforms = "loudness", None
    assert next(iter(dataset.get_loader(batch_size=3, shuffle=False, batch_features="loudness", batch_transforms=batch_transforms))).shape == (3,)
    # transforms of batch features are applied after their computation
    dataset.output_pattern, dataset.transforms = "{loudness->l}", {"l": Half()}
    l_half = next(iter(dataset.get_loader(batch_size=2, shuffle=False, batch_features="loudness", batch_transforms=batch_transforms)))["l"]
    dataset.transforms = None
    l = next(iter(dataset.get_loader(batch_size=2, shuffle=False, batch_features="loudness")))["l"]
    assert torch.allclose(l_half, l * 0.5)


def test_workers(paths):
    feature = Loudness(sr=SR, keep_channels=False)
    dataset = online(paths, features=[feature, Width()], output_pattern="{waveform->x,loudness,width}", transforms={"x": Half()})
    loader = dataset.get_loader(batch_size=2, shuffle=False, batch_features=["loudness"], num_workers=2, multiprocessing_context="spawn")
    assert isinstance(loader, BatchFeatureLoader) and len(loader) == len(dataset) // 2
    # workers do not compute batch features
    assert [f.feature_name for f in loader.dataset.features] == ["width"]
    assert [f.feature_name for f in dataset.features] == ["loudness", "width"]
    batch = next(iter(loader))
    assert set(batch.keys()) == {"x", "loudness", "width"}
    # batch features on accelerators do not force spawn, as workers do not compute them
    feature.device = torch.device("cuda")
    loader = dataset.get_loader(batch_size=2, batch_features="auto", num_workers=1)
    assert [f.feature_name for f in loader.processor.features] == ["loudness"]
    assert loader.loader.multiprocessing_context is None


def test_preprocessed(paths):
    dataset = AudioDataset(paths["pre"], output_pattern="{waveform,loudness,width}")
    # pre-processed features are read ; new features can be computed on batches
    with pytest.raises(ValueError):
        dataset.get_loader(batch_size=2, batch_features=["loudness"])
    batch = next(iter(dataset.get_loader(batch_size=2, shuffle=False, batch_features=[Width()])))
    assert batch["width"].shape == (2, (SR // 2) // 2048)
    stored = AudioDataset(paths["pre"], output_pattern="loudness")
    assert torch.allclose(batch["loudness"], stored[0:2])


def test_combined(paths):
    combined = CombinedAudioDataset([AudioDataset(paths["pre"], output_pattern="{waveform,width}"),
                                     online(paths, channels=2, features=[Width()], output_pattern="{waveform,width}")])
    loader = combined.get_loader(batch_size=4, batch_features=[Width()], drop_last=True)
    batch = next(iter(loader))
    assert batch["waveform"].shape == (4, 2, SR // 2) and batch["width"].shape == (4, (SR // 2) // 2048)


def test_compute_batch_features(paths):
    dataset = online(paths, features=features())
    audio = 0.1 * torch.randn(3, 1, SR // 2)
    out = dataset.compute_batch_features(audio)
    assert set(out.keys()) == {"loudness", "mel_128", "width", "emb"}
    assert out["loudness"].shape == (3,) and out["emb"].shape[:2] == (3, 1)
    assert set(dataset.compute_batch_features(audio[:, 0], features=["loudness", Width(name="ms")]).keys()) == {"loudness", "ms"}
    with pytest.raises(ValueError):
        dataset.compute_batch_features(audio, features=["unknown"])


def test_numpy_transforms_on_tensors():
    # audiomentations transforms take tensors from any device (e.g. batches on GPU), and give them back on this device
    gain = adt.Gain(sr=SR)
    x = (0.1 * torch.randn(2, 1, 4096)).requires_grad_(True)
    y = gain(x, _force_transform=True)
    assert torch.is_tensor(y) and y.shape == x.shape and y.device == x.device


class RandomGain():
    """random (picklable) augmentation : features must be computed on the drawn gain."""
    def __call__(self, x):
        return x * float(torch.empty(1).uniform_(0.1, 1.))


@pytest.mark.parametrize("transform", [Half(), RandomGain()])
def test_item_features_after_transforms(paths, transform):
    """features computed item by item are also computed on augmented audio."""
    loudness = Loudness(sr=SR, keep_channels=False)
    dataset = online(paths, features=[Loudness(sr=SR, keep_channels=False)], output_pattern="{waveform->x,loudness}", transforms={"x": transform})
    for i in range(4):
        item = dataset[i]
        assert torch.allclose(torch.as_tensor(item["loudness"]), loudness(item["x"]), atol=1e-4)
    # batches of the data loader
    batch = next(iter(dataset.get_loader(batch_size=4, shuffle=False)))
    expected = torch.stack([loudness(batch["x"][i]) for i in range(4)])
    assert torch.allclose(batch["loudness"], expected, atol=1e-4)


@pytest.mark.parametrize("batch_features", [None, ["loudness"]])
def test_clean_and_augmented_entries(paths, batch_features):
    """with clean and augmented waveforms, features are computed on augmented audio, and clean audio is kept."""
    loudness = Loudness(sr=SR, keep_channels=False)
    dataset = online(paths, features=[Loudness(sr=SR, keep_channels=False)], output_pattern="{waveform->clean,waveform->x,loudness}", 
                     transforms={"x": Half()})
    batch = next(iter(dataset.get_loader(batch_size=4, shuffle=False, batch_features=batch_features)))
    assert torch.allclose(batch["x"], batch["clean"] * 0.5, atol=1e-6)
    expected = torch.stack([loudness(batch["x"][i]) for i in range(4)])
    assert torch.allclose(batch["loudness"], expected, atol=1e-4)


def test_worker_features_with_batch_transforms(paths):
    """with transforms applied on batches, features computed in workers would miss augmentations : they are computed on
    batches."""
    loudness = Loudness(sr=SR, keep_channels=False)
    dataset = online(paths, features=[Loudness(sr=SR, keep_channels=False), Mel(sr=SR, n_fft=1024)], 
                     output_pattern="{waveform->x,loudness,mel_128}", transforms={"x": Half()})
    loader = dataset.get_loader(batch_size=4, shuffle=False, batch_features=["mel_128"], batch_transforms=True)
    assert sorted(f.feature_name for f in loader.processor.features) == ["loudness", "mel_128"]
    batch = next(iter(loader))
    expected = torch.stack([loudness(batch["x"][i]) for i in range(4)])
    assert torch.allclose(batch["loudness"], expected, atol=1e-4)


@pytest.mark.parametrize("batch_features,batch_transforms", [(None, False), (["loudness"], False), (["loudness"], True), (None, True)])
def test_features_before_transforms(paths, batch_features, batch_transforms):
    """with features_after_transforms=False, features describe loaded audio, before transforms."""
    loudness = Loudness(sr=SR, keep_channels=False)
    dataset = online(paths, features=[Loudness(sr=SR, keep_channels=False)], output_pattern="{waveform->x,loudness}",
                     transforms={"x": Half()}, features_after_transforms=False)
    assert not dataset.features_after_transforms
    for i in range(2):
        item = dataset[i]
        assert torch.allclose(torch.as_tensor(item["loudness"]), loudness(item["x"] * 2), atol=1e-4)
    batch = next(iter(dataset.get_loader(batch_size=4, shuffle=False, batch_features=batch_features, batch_transforms=batch_transforms)))
    expected = torch.stack([loudness(batch["x"][i] * 2) for i in range(4)])
    assert torch.allclose(batch["loudness"], expected, atol=1e-4)


def test_features_before_transforms_with_clean_entry(paths):
    """batch features computed before transforms use the clean waveform entry if any."""
    loudness = Loudness(sr=SR, keep_channels=False)
    dataset = online(paths, features=[Loudness(sr=SR, keep_channels=False)], output_pattern="{waveform->clean,waveform->x,loudness}",
                     transforms={"x": Half()}, features_after_transforms=False)
    loader = dataset.get_loader(batch_size=4, shuffle=False, batch_features=["loudness"])
    assert loader.processor.audio_entry == 0
    batch = next(iter(loader))
    expected = torch.stack([loudness(batch["clean"][i]) for i in range(4)])
    assert torch.allclose(batch["loudness"], expected, atol=1e-4)


def test_configure_features_after_transforms(paths):
    """features_after_transforms can be set with configure, get_loader or the property, without modifying the dataset."""
    loudness = Loudness(sr=SR, keep_channels=False)
    dataset = online(paths, features=[Loudness(sr=SR, keep_channels=False)], output_pattern="{waveform->x,loudness}", transforms={"x": Half()})
    assert dataset.features_after_transforms
    before = dataset.configure(features_after_transforms=False)
    assert not before.features_after_transforms and dataset.features_after_transforms
    item = before[0]
    assert torch.allclose(torch.as_tensor(item["loudness"]), loudness(item["x"] * 2), atol=1e-4)
    batch = next(iter(dataset.get_loader(batch_size=2, shuffle=False, features_after_transforms=False)))
    assert torch.allclose(batch["loudness"], torch.stack([loudness(batch["x"][i] * 2) for i in range(2)]), atol=1e-4)
    assert dataset.features_after_transforms
    dataset.features_after_transforms = False
    item = dataset[0]
    assert torch.allclose(torch.as_tensor(item["loudness"]), loudness(item["x"] * 2), atol=1e-4)


def test_combined_features_after_transforms(paths):
    loudness = Loudness(sr=SR, keep_channels=False)
    kwargs = dict(features=[Loudness(sr=SR, keep_channels=False)], output_pattern="{waveform->x,loudness}", transforms={"x": Half()})
    datasets = [online(paths, **kwargs), online(paths, **kwargs)]
    combined = CombinedAudioDataset(datasets, features_after_transforms=False)
    # given datasets are not modified
    assert all(d.features_after_transforms for d in datasets)
    assert not combined.features_after_transforms and not any(d.features_after_transforms for d in combined.datasets)
    item = combined[0]
    assert torch.allclose(torch.as_tensor(item["loudness"]), loudness(item["x"] * 2), atol=1e-4)
    configured = combined.configure(features_after_transforms=True)
    assert configured.features_after_transforms and all(d.features_after_transforms for d in configured.datasets)
    item = configured[0]
    assert torch.allclose(torch.as_tensor(item["loudness"]), loudness(item["x"]), atol=1e-4)
    combined.features_after_transforms = True
    assert all(d.features_after_transforms for d in combined.datasets)
