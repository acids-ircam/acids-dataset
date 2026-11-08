"""devices of features : CPU by default, accelerators on request ("auto" for the default accelerated device)."""
import pytest
import torch

from acids_dataset.utils import env_flag, resolve_device, get_default_accelerated_device
from acids_dataset.features import Loudness, ModuleEmbedding

SR = 22050
MODULE_PATH = "tests/scripted/conv_embedding.ts"


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("yes", True), ("0", False), ("false", False),
                                            ("No", False), ("off", False), ("", False)])
def test_env_flag(monkeypatch, value, expected):
    monkeypatch.setenv("ACIDS_TEST_FLAG", value)
    assert env_flag("ACIDS_TEST_FLAG", not expected) == expected


def test_env_flag_default(monkeypatch):
    monkeypatch.delenv("ACIDS_TEST_FLAG", raising=False)
    assert env_flag("ACIDS_TEST_FLAG", True) and not env_flag("ACIDS_TEST_FLAG", False)


def test_resolve_device():
    assert resolve_device(None) == torch.device("cpu")
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")
    assert resolve_device("auto") == get_default_accelerated_device()


def test_default_accelerated_device(monkeypatch):
    # USE_CUDA=0 and USE_MPS=0 disable accelerators (environment variables were read as truthy strings)
    monkeypatch.setenv("USE_CUDA", "0")
    monkeypatch.setenv("USE_MPS", "0")
    assert get_default_accelerated_device() == torch.device("cpu")
    monkeypatch.delenv("USE_MPS")
    assert get_default_accelerated_device() == torch.device("cpu")
    monkeypatch.delenv("USE_CUDA")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert get_default_accelerated_device().type == expected


def test_features_default_to_cpu():
    assert Loudness(sr=SR).device == torch.device("cpu")
    feature = ModuleEmbedding(module_path=MODULE_PATH, sr=SR)
    assert feature.device == torch.device("cpu")
    assert all(p.device.type == "cpu" for p in feature._module.parameters())
    assert feature.to("auto").device == get_default_accelerated_device()
    assert feature.to(None).device == torch.device("cpu")


@pytest.mark.skipif(not (torch.cuda.is_available() or torch.backends.mps.is_available()), reason="no accelerator")
def test_batch_call_device():
    device = "cuda" if torch.cuda.is_available() else "mps"
    feature = ModuleEmbedding(module_path=MODULE_PATH, sr=SR, device=device)
    x = 0.1 * torch.randn(2, 1, SR // 2)
    # embeddings are given on the device of inputs, as with __call__
    assert feature.batch_call(x).device.type == "cpu"
    assert feature.batch_call(x.to(device)).device.type == device
