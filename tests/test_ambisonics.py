"""Ambisonic encoding and decoding."""
import math
import numpy as np
import pytest
import torch
from acids_dataset.features import AmbisonicEncoder, AmbisonicDecoder
from acids_dataset.features.ambisonics import spherical_harmonics, fibonacci_sphere, get_layout, LAYOUTS


def test_spherical_harmonics_sn3d():
    az, el = math.radians(60), math.radians(20)
    y = spherical_harmonics(2, 60, 20)[0]
    s3 = math.sqrt(3) / 2
    expected = [1, math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el),
                s3 * math.sin(2 * az) * math.cos(el) ** 2, s3 * math.sin(az) * math.sin(2 * el), (3 * math.sin(el) ** 2 - 1) / 2,
                s3 * math.cos(az) * math.sin(2 * el), s3 * math.cos(2 * az) * math.cos(el) ** 2]
    assert np.allclose(y, expected)


def test_spherical_harmonics_n3d_orthonormal():
    directions = fibonacci_sphere(20000)
    y = spherical_harmonics(3, directions[:, 0], directions[:, 1], "n3d")
    assert np.allclose(y.T @ y / len(directions), np.eye(16), atol=2e-3)


@pytest.mark.parametrize("n_channels, order", [(1, 0), (2, 1), (3, 1), (4, 2), (5, 2), (7, 3)])
def test_adaptive_order(n_channels, order):
    encoder = AmbisonicEncoder()
    assert encoder.get_order(n_channels) == order
    assert encoder(torch.randn(n_channels, 100)).shape == ((order + 1) ** 2, 100)


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_roundtrip(layout):
    n = len(get_layout(layout))
    x = torch.randn(3, n, 500)
    b = AmbisonicEncoder(layout=layout)(x)
    assert torch.allclose(AmbisonicDecoder(layout=layout)(b), x, atol=1e-4)


def test_fixed_orders():
    x = torch.randn(4, 2, 100)
    assert AmbisonicEncoder(order=3)(x).shape == (4, 16, 100)
    b = AmbisonicEncoder(order=3, layout=[(10, 5)])(torch.randn(1, 100))
    assert AmbisonicDecoder()(b).shape == (16, 100)                      # adaptive : 16 virtual loudspeakers
    assert AmbisonicDecoder(order=3, layout="cube")(b).shape == (8, 100)
    # fixed orders need (N + 1)^2 channels : higher orders are not discarded
    with pytest.raises(ValueError):
        AmbisonicDecoder(order=1, layout="cube")(b)
    # fixed orders must encode channels invertibly
    with pytest.raises(ValueError):
        AmbisonicEncoder(order=1, layout="7.0")(torch.randn(7, 100))
    assert AmbisonicEncoder(order=3, layout="7.0")(torch.randn(7, 100)).shape == (16, 100)


def test_decoding_direction():
    # a plane wave is decoded the loudest on the closest loudspeakers
    for method in AmbisonicDecoder.methods:
        b = AmbisonicEncoder(order=2, layout=[(45, 0)])(torch.ones(1, 10))
        gains = AmbisonicDecoder(layout="octagon", method=method)(b)[:, 0]
        assert int(gains.argmax()) == 1


def test_errors():
    with pytest.raises(ValueError):
        AmbisonicDecoder()(torch.randn(5, 10))              # not (N + 1)^2 channels
    with pytest.raises(ValueError):
        AmbisonicDecoder(order=2)(torch.randn(4, 10))
    with pytest.raises(ValueError):
        AmbisonicEncoder(layout="stereo")(torch.randn(3, 10))
    with pytest.raises(ValueError):
        AmbisonicEncoder(layout=[(0, 0), (0, 0)])(torch.randn(2, 10))
    with pytest.raises(ValueError):
        AmbisonicEncoder(order=-1)
    with pytest.raises(ValueError):
        AmbisonicEncoder(layout="unknown")(torch.randn(2, 10))
