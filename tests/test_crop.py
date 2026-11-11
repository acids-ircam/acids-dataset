"""Crop : random crops, and padding of shorter signals."""
import numpy as np
import pytest
import torch

from acids_dataset.transforms import Crop


def test_crop():
    x = torch.arange(100.).expand(2, 3, 100)
    y = Crop(n_signal=10)(x)
    assert y.shape == (2, 3, 10) and torch.equal(y[..., 1:] - y[..., :-1], torch.ones(2, 3, 9))
    assert Crop(n_signal=2, dim=1)(x).shape == (2, 2, 100)
    assert Crop()(x) is x


@pytest.mark.parametrize("mode", ["zero", "repeat", "replicate", "reflect"])
@pytest.mark.parametrize("pad_dir", ["right", "left", "center"])
def test_crop_pad(mode, pad_dir):
    x = torch.arange(1., 6.).expand(2, 5)
    y = Crop(n_signal=12, pad_if_shorter=mode, pad_dir=pad_dir)(x)
    assert y.shape == (2, 12)
    start = {"right": 0, "left": 7, "center": 3}[pad_dir]
    assert torch.equal(y[:, start:start + 5], x)
    if mode == "zero":
        assert y.abs().sum() == x.abs().sum()


def test_crop_pad_dim_and_numpy():
    x = torch.randn(2, 3, 50)
    y = Crop(n_signal=5, dim=1)(x)
    assert y.shape == (2, 5, 50) and torch.equal(y[:, :3], x) and torch.all(y[:, 3:] == 0)
    z = Crop(n_signal=80)(x.numpy())
    assert isinstance(z, np.ndarray) and z.shape == (2, 3, 80)


def test_crop_no_pad():
    with pytest.raises(ValueError):
        Crop(n_signal=80, pad_if_shorter=None)(torch.randn(2, 50))
