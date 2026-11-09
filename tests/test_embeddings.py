"""Neural embeddings in online datasets : pickling, TorchScript modules in data loader workers, devices."""
import pickle
import pytest
import torch
import torchaudio

from acids_dataset.datasets import AudioDataset
from acids_dataset.features import ModuleEmbedding, Loudness

SR = 22050
TS_PATH = "tests/scripted/conv_embedding.ts"


class Downsample(torch.nn.Module):
    def forward(self, x):
        return x[..., ::64]


@pytest.fixture
def raw(tmp_path):
    for i in range(4):
        torchaudio.save(str(tmp_path / f"{i}.wav"), 0.1 * torch.randn(1, 2 * SR), sample_rate=SR)
    return tmp_path


def test_pickling_keeps_module():
    feature = ModuleEmbedding(module=Downsample(), sr=SR, name="emb")
    copy = pickle.loads(pickle.dumps(feature))
    # pickling does not modify the feature itself
    assert feature._module is not None
    assert copy._module is None
    retained = pickle.loads(pickle.dumps(ModuleEmbedding(module=Downsample(), sr=SR, name="emb", retain_module=True)))
    assert retained(torch.randn(1, 1, 640)).shape[-1] == 10


def test_torchscript_reload():
    feature = ModuleEmbedding(module_path=TS_PATH, sr=SR)
    copy = pickle.loads(pickle.dumps(feature))
    x = torch.randn(1, 1, 4096)
    assert torch.allclose(copy(x), feature(x))


@pytest.mark.parametrize("module", ["torchscript", "retained"])
def test_embeddings_in_workers(raw, module):
    if module == "torchscript":
        feature = ModuleEmbedding(module_path=TS_PATH, sr=SR, name="emb")
    else:
        feature = ModuleEmbedding(module=Downsample(), sr=SR, name="emb", retain_module=True)
    dataset = AudioDataset(raw, sr=SR, chunk_length=8192, features=[feature], output_pattern="(waveform,emb)", write_index=False)
    loader = dataset.get_loader(batch_size=2, num_workers=2, multiprocessing_context="spawn", drop_last=True)
    waveform, embedding = next(iter(loader))
    assert waveform.shape == (2, 1, 8192) and embedding.shape[0] == 2
    # the module of the main process is kept
    assert feature._module is not None


def test_to_and_spawn(raw):
    feature = Loudness(sr=SR)
    dataset = AudioDataset(raw, sr=SR, chunk_length=8192, features=[feature], write_index=False)
    assert dataset.to("cpu") is dataset and dataset.feature_devices == {"cpu"}
    assert dataset.get_loader(batch_size=2, num_workers=1).multiprocessing_context is None
    # features on accelerators : workers are started with spawn (CUDA cannot be used in forked processes)
    feature.device = torch.device("cuda")
    assert dataset.feature_devices == {"cuda"}
    loader = dataset.get_loader(batch_size=2, num_workers=1)
    assert loader.multiprocessing_context.get_start_method() == "spawn"
    assert dataset.get_loader(batch_size=2, num_workers=1, multiprocessing_context="fork").multiprocessing_context.get_start_method() == "fork"


def test_module_to():
    feature = ModuleEmbedding(module=Downsample(), sr=SR, name="emb")
    assert feature.to("cpu") is feature and feature.device == torch.device("cpu")
