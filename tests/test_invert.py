"""invertible features (invertible = True) give audio back from their outputs with invert."""
import math
import pytest
import torch

from acids_dataset.features import (AcidsDatasetFeature, STFT, DGT, FFT, DCT, PQMF, MidSide, AmbisonicEncoder,
                                    AmbisonicDecoder, Loudness, Width, Mel)

SR = 22050


def signal(channels=2, length=SR, seed=0):
    return torch.randn(3, channels, length, generator=torch.Generator().manual_seed(seed))


def snr(estimate, target):
    return float(10 * torch.log10(target.square().sum() / (estimate - target).square().sum()))


def test_invertible_flags():
    for feature in [STFT(sr=SR), DGT(sr=SR), FFT(sr=SR), DCT(sr=SR), PQMF(n_bands=4, sr=SR), MidSide(sr=SR),
                    AmbisonicEncoder(sr=SR), AmbisonicDecoder(sr=SR)]:
        assert feature.invertible, feature
    for feature in [Loudness(sr=SR), Width(sr=SR), Mel(sr=SR)]:
        assert not feature.invertible
        with pytest.raises(NotImplementedError):
            feature.invert(torch.zeros(1, 10))


# STFT

@pytest.mark.parametrize("kwargs", [dict(mode="complex", center=True), dict(mode=["magnitude", "phase"], keep_channels=True, center=True),
                                    dict(mode=["real", "imaginary"], center=True)])
def test_stft_length(kwargs):
    x = signal(length=SR + 37)
    feature = STFT(frame_length=1024, hop_length=256, sr=SR, **kwargs)
    y = feature.invert(feature(x), length=x.shape[-1])
    target = x if feature.keep_channels else x.mean(-2)
    assert y.shape == target.shape
    n = (x.shape[-1] // 256) * 256
    assert torch.allclose(y[..., :n], target[..., :n], atol=1e-4)


# DGT

@pytest.mark.parametrize("kwargs", [dict(mode="complex"), dict(mode=["magnitude", "phase"], keep_channels=True),
                                    dict(mode=["real", "imaginary"]), dict(mode=["db", "phase"], keep_channels=True),
                                    dict(mode=["power", "phase"])])
def test_dgt_inversion(kwargs):
    x = signal()
    feature = DGT(n_fft=1024, hop_length=256, sr=SR, **kwargs)
    y = feature(x)
    n_modes = len(feature.mode)
    assert y.shape[-2:] == (513, SR // 256 + 1)
    assert y.shape[:-2] == ((3, 2 * n_modes) if feature.keep_channels else ((3, n_modes) if n_modes > 1 else (3,)))
    target = x if feature.keep_channels else x.mean(-2)
    inverted = feature.invert(y, length=SR)
    assert inverted.shape == target.shape
    # samples after the last hop are not covered by frames
    n = (SR // 256) * 256
    assert snr(inverted[..., :n], target[..., :n]) > 60 and (inverted[..., n:] == 0).all()


def test_dgt_magnitude_inversion():
    # sum of sinusoids : PGHI reconstructs phases from magnitudes
    t = torch.arange(SR) / SR
    x = sum(torch.sin(2 * math.pi * f * t + p) for f, p in [(220, 0.), (440, 1.), (1250, 2.)])[None]
    feature = DGT(n_fft=1024, hop_length=256, sr=SR)
    magnitude = feature(x)
    for inversion_mode in [None, "griffin_lim", "random"]:
        y = feature.invert(magnitude, inversion_mode=inversion_mode, length=SR)
        assert y.shape == (SR,)
    y = feature.invert(magnitude, length=SR)
    assert (feature(y) - magnitude).norm() / magnitude.norm() < 0.1


def test_dgt_pickling():
    import pickle
    feature = DGT(n_fft=512, mode="complex", sr=SR)
    x = signal(length=4096)
    restored = pickle.loads(pickle.dumps(feature))
    assert torch.allclose(restored(x), feature(x))
    assert torch.allclose(restored.invert(restored(x), length=4096), x.mean(-2), atol=1e-4)


# FFT

@pytest.mark.parametrize("kwargs", [dict(mode="complex"), dict(mode=["magnitude", "phase"], keep_channels=True),
                                    dict(mode=["real", "imaginary"], onesided=False), dict(mode=["db", "phase"], window="hamming"),
                                    dict(mode=["power_density", "phase"], keep_channels=True, window="hamming"),
                                    dict(mode="complex", n_fft=32768)])
@pytest.mark.parametrize("length", [SR, SR + 1])
def test_fft_inversion(kwargs, length):
    x = signal(length=length)
    feature = FFT(sr=SR, **kwargs)
    y = feature.invert(feature(x), length=length)
    target = x if feature.keep_channels else x.mean(-2)
    assert y.shape == target.shape
    assert snr(y, target) > 60


def test_fft_cropped():
    x = signal()
    feature = FFT(n_fft=4096, mode="complex", keep_channels=True, sr=SR)
    y = feature.invert(feature(x), length=SR)
    assert torch.allclose(y[..., :4096], x[..., :4096], atol=1e-4) and (y[..., 4096:] == 0).all()


def test_fft_without_phase():
    feature = FFT(sr=SR)
    with pytest.raises(ValueError):
        feature.invert(feature(signal()))


# DCT

@pytest.mark.parametrize("kwargs", [dict(window=None), dict(window=None, hop_length=512, keep_channels=True),
                                    dict(window="hann", hop_length=512), dict(window="hamming", hop_length=256, keep_channels=True)])
def test_dct_inversion(kwargs):
    x = signal(length=64 * 1024)
    feature = DCT(frame_length=1024, sr=SR, **kwargs)
    y = feature.invert(feature(x), length=x.shape[-1])
    target = x if feature.keep_channels else x.mean(-2)
    assert y.shape == target.shape
    # with hann windows, borders of signals are only covered by vanishing window borders
    border = feature.hop_length if feature.window == "hann" else 0
    n = x.shape[-1] - border
    assert torch.allclose(y[..., border:n], target[..., border:n], atol=1e-4)


def test_dct_truncated():
    # fewer coefficients : frames are low-passed in the DCT domain
    t = torch.arange(8192) / SR
    x = torch.sin(2 * math.pi * 100 * t)[None, None]
    feature = DCT(frame_length=1024, hop_length=512, n_coeffs=256, keep_channels=True, sr=SR)
    assert snr(feature.invert(feature(x), length=8192)[..., 512:-512], x[..., 512:-512]) > 20


def test_dct_tail():
    # samples after the last frame are lost
    x = signal(length=2500)
    feature = DCT(frame_length=1024, window=None, keep_channels=True, sr=SR)
    y = feature.invert(feature(x), length=2500)
    assert torch.allclose(y[..., :2048], x[..., :2048], atol=1e-4) and (y[..., 2048:] == 0).all()


# PQMF

@pytest.mark.parametrize("n_bands", [2, 4, 16])
def test_pqmf_inversion(n_bands):
    x = signal(length=SR)
    feature = PQMF(n_bands=n_bands, sr=SR)
    bands = feature(x)
    y = feature.invert(bands, length=SR)
    assert y.shape == x.shape
    # near perfect reconstruction, outside of filter borders
    border = feature.filter_length
    assert snr(y[..., border:-border], x[..., border:-border]) > 50
    # with inverse=True, invert gives sub-bands back
    synthesis = PQMF(n_bands=n_bands, inverse=True, sr=SR)
    assert torch.allclose(synthesis.invert(synthesis(bands)), feature(synthesis(bands)))


# spatial

@pytest.mark.parametrize("normalization", ["average", "orthonormal"])
@pytest.mark.parametrize("inverse", [False, True])
def test_midside_inversion(normalization, inverse):
    x = signal()
    feature = MidSide(normalization=normalization, inverse=inverse, sr=SR)
    assert torch.allclose(feature.invert(feature(x)), x, atol=1e-6)


@pytest.mark.parametrize("layout,channels", [(None, 1), (None, 2), (None, 3), (None, 4), (None, 5), (None, 7),
                                             ("tetrahedron", 4), ("5.0", 5), ("7.0", 7), ("cube", 8)])
def test_ambisonic_encoder_inversion(layout, channels):
    """the order is inferred from the number of ambisonic channels, and the number of channels from the layout, or
    from signals with layout=None (e.g. stereo and 3 channels both give first order ambisonics)."""
    x = signal(channels=channels, length=4096)
    feature = AmbisonicEncoder(layout=layout, sr=SR)
    y = feature.invert(feature(x))
    assert y.shape == x.shape and torch.allclose(y, x, atol=1e-4)
    assert torch.allclose(feature.invert(feature(x), n_channels=channels), x, atol=1e-4)


@pytest.mark.parametrize("order,channels", [(1, 2), (2, 3), (3, 7)])
def test_ambisonic_encoder_fixed_order(order, channels):
    x = signal(channels=channels, length=4096)
    feature = AmbisonicEncoder(order=order, sr=SR)
    a = feature(x)
    assert a.shape[-2] == (order + 1) ** 2
    assert torch.allclose(feature.invert(a), x, atol=1e-4)
    # ambisonic signals of another order
    with pytest.raises(ValueError):
        feature.invert(AmbisonicEncoder(order=order + 1, sr=SR)(x))


def test_ambisonic_encoder_errors():
    feature = AmbisonicEncoder(sr=SR)
    with pytest.raises(ValueError):
        feature.invert(signal(channels=5))                                  # not (N + 1)^2 channels
    with pytest.raises(ValueError):
        feature.invert(feature(signal(channels=2)), n_channels=7)           # 7 channels give order 3
    with pytest.raises(ValueError):
        AmbisonicEncoder(order=1, sr=SR)(signal(channels=4))                # 4 horizontal channels need order 2


@pytest.mark.parametrize("kwargs,order", [(dict(), 1), (dict(), 3), (dict(layout="cube", order=1), 1),
                                          (dict(method="sampling"), 2), (dict(layout="cube"), 1), (dict(layout="cube"), 0),
                                          (dict(layout=[(az, el) for el in (-45, 0, 45) for az in range(0, 360, 40)]), 2)])
def test_ambisonic_decoder_inversion(kwargs, order):
    """the order is the fixed order, or is inferred from the number of loudspeakers (and from signals for layouts
    that can hold several orders, e.g. order 0 or 1 for cubes)."""
    a = signal(channels=(order + 1) ** 2, length=4096)
    feature = AmbisonicDecoder(sr=SR, **kwargs)
    speakers = feature(a)
    y = feature.invert(speakers)
    assert y.shape == a.shape and torch.allclose(y, a, atol=1e-3)
    assert torch.allclose(feature.invert(speakers, order=order), a, atol=1e-3)


def test_ambisonic_decoder_errors():
    with pytest.raises(ValueError):
        AmbisonicDecoder(order=1, sr=SR)(signal(channels=9))               # fixed order : 4 channels
    with pytest.raises(ValueError):
        AmbisonicDecoder(order=1, sr=SR).invert(signal(channels=9))        # 4 virtual loudspeakers
    with pytest.raises(ValueError):
        AmbisonicDecoder(layout="cube", sr=SR).invert(signal(channels=5))  # 8 loudspeakers


def test_ambisonic_decoder_horizontal():
    # horizontal layouts lose vertical components : only horizontal components are recovered
    a = signal(channels=4, length=4096)
    feature = AmbisonicDecoder(layout="octagon", order=1, sr=SR)
    y = feature.invert(feature(a))
    horizontal = [0, 1, 3]       # W, Y, X (ACN)
    assert torch.allclose(y[..., horizontal, :], a[..., horizontal, :], atol=1e-4)
    assert not torch.allclose(y[..., 2, :], a[..., 2, :], atol=1e-2)


@pytest.mark.parametrize("feature", [STFT(mode="complex", sr=SR), DGT(mode="complex", sr=SR), FFT(mode="complex", sr=SR),
                                     DCT(window=None, sr=SR), PQMF(n_bands=4, sr=SR), MidSide(sr=SR)])
def test_invert_batches_and_devices(feature):
    """inversions keep the batch shape and the device of inputs."""
    x = signal(channels=2, length=8192)[:, None]           # (3, 1, 2, T)
    y = feature(x)
    inverted = feature.invert(y, length=8192) if "length" in feature.invert.__code__.co_varnames else feature.invert(y)
    assert inverted.shape[:2] == (3, 1) and inverted.device == x.device
