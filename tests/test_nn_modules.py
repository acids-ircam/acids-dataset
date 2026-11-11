"""nn_module() of transforms and features : modules (and their TorchScript versions) must give the outputs of the
transforms / features they come from, as well as their inversions."""
import math
import pytest
import torch

from acids_dataset import transforms as T
from acids_dataset import features as F
# transforms.Resample is the audiomentations one : the torchaudio-based Resample is in transforms.misc
from acids_dataset.transforms.misc import Resample as MiscResample, Normalize as MiscNormalize, Gain as MiscGain

SR = 44100
N_SAMPLES = 22050


def _signal(n_channels: int, batch: int = 2, n_samples: int = N_SAMPLES) -> torch.Tensor:
    """harmonic tone + noise bursts, with a silent part and a few clipped samples."""
    torch.manual_seed(0)
    t = torch.arange(n_samples) / SR
    f0 = torch.tensor([220., 330.])[:batch, None, None] * (1 + torch.arange(n_channels)[None, :, None] * 0.01)
    tone = sum(torch.sin(2 * math.pi * k * f0 * t) / k for k in range(1, 6))
    envelope = (torch.sin(2 * math.pi * 4 * t) > 0).float()
    x = 0.5 * tone + 0.3 * torch.randn(batch, n_channels, n_samples) * envelope
    x[..., -n_samples // 8:] = 0.
    return (1.2 * x).clamp(-1, 1)


def _close(a, b, rtol=1e-4, atol=1e-4):
    torch.testing.assert_close(torch.as_tensor(a).to(torch.as_tensor(b).dtype), torch.as_tensor(b), rtol=rtol, atol=atol)


def _modules(obj, **kwargs):
    module = obj.nn_module(**kwargs)
    assert isinstance(module, torch.nn.Module)
    return module, torch.jit.script(module)


# --------------------------------------------------------------------------------------------------------------------- #
#  transforms
# --------------------------------------------------------------------------------------------------------------------- #

# (transform, input channels, nn_module kwargs, invert kwargs of the transform, or None if not invertible)
TRANSFORMS = [
    (lambda: T.MidSide(), 2, {}, {}),
    (lambda: T.MidSide(normalization="orthonormal"), 2, {}, {}),
    (lambda: T.MidSide(inverse=True), 2, {}, {}),
    (lambda: T.AmbisonicEncoder(), 2, {"n_channels": 2}, {"n_channels": 2}),
    (lambda: T.AmbisonicEncoder(), 4, {"n_channels": 4}, {"n_channels": 4}),
    (lambda: T.AmbisonicEncoder(layout="5.0"), 5, {}, {"n_channels": 5}),
    (lambda: T.AmbisonicDecoder(), 4, {"n_channels": 4}, {"order": 1}),
    (lambda: T.AmbisonicDecoder(layout="5.0", order=1), 4, {}, {"order": 1}),
    (lambda: T.AmbisonicDecoder(order=1, method="sampling"), 4, {}, {"order": 1}),
    (lambda: MiscResample(sr=SR, target_sr=22050), 1, {}, None),
    (lambda: MiscResample(sr=SR, target_sr=48000), 2, {}, None),
    (lambda: T.PreEmphasis(), 1, {}, None),
    (lambda: T.Log1p(), 1, {}, None),
    (lambda: T.Exp1m(), 1, {}, None),
    (lambda: MiscNormalize(), 2, {}, None),
    (lambda: T.Derivator(), 1, {}, None),
    (lambda: T.Integrator(sr=SR), 1, {}, None),
    (lambda: T.NormRange(), 2, {}, None),
    (lambda: T.NormRange(max_value=0.5, boost=False), 2, {}, None),
    (lambda: T.Compose(T.Log1p(), T.Exp1m()), 1, {}, None),
]


@pytest.mark.parametrize("make, n_channels, module_kwargs, invert_kwargs", TRANSFORMS,
                         ids=[str(i) for i in range(len(TRANSFORMS))])
def test_transform_nn_module(make, n_channels, module_kwargs, invert_kwargs):
    transform = make()
    x = _signal(n_channels)
    expected = transform(x.clone(), _force_transform=True)
    module, scripted = _modules(transform, **module_kwargs)
    for m in (module, scripted):
        _close(m(x.clone()), expected)
    if invert_kwargs is not None:
        expected_inverse = transform.invert(expected.clone(), **invert_kwargs)
        for m in (module, scripted):
            _close(m.invert(expected.clone()), expected_inverse)


@pytest.mark.parametrize("transform", [T.Stretch(sr=SR), T.Gain(), T.RandomDelay(), T.AddNoise()])
def test_transform_without_nn_module(transform):
    with pytest.raises(NotImplementedError):
        transform.nn_module()


def test_ambisonic_modules_need_channels():
    with pytest.raises(ValueError):
        T.AmbisonicEncoder().nn_module()
    with pytest.raises(ValueError):
        T.AmbisonicDecoder().nn_module()


# --------------------------------------------------------------------------------------------------------------------- #
#  features
# --------------------------------------------------------------------------------------------------------------------- #

FEATURES = [
    lambda k: F.RMS(keep_channels=k),
    lambda k: F.RMS(db=False, hop_length=512, keep_channels=k),
    lambda k: F.ZeroCrossingRate(keep_channels=k),
    lambda k: F.ZeroCrossingRate(unit="hz", keep_channels=k),
    lambda k: F.AttackTime(keep_channels=k),
    lambda k: F.TemporalCentroid(keep_channels=k),
    lambda k: F.CrestFactor(keep_channels=k),
    lambda k: F.CrestFactor(db=False, keep_channels=k),
    lambda k: F.SpectralCentroid(keep_channels=k),
    lambda k: F.SpectralCentroid(unit="midi", keep_channels=k),
    lambda k: F.SpectralSpread(unit="normalized", keep_channels=k),
    lambda k: F.SpectralSkewness(keep_channels=k),
    lambda k: F.SpectralKurtosis(keep_channels=k),
    lambda k: F.SpectralRolloff(keep_channels=k),
    lambda k: F.SpectralFlatness(keep_channels=k),
    lambda k: F.SpectralFlatness(db=True, keep_channels=k),
    lambda k: F.SpectralCrest(keep_channels=k),
    lambda k: F.SpectralFlux(hop_length=512, keep_channels=k),
    lambda k: F.SpectralContrast(keep_channels=k),
    lambda k: F.ClippingRatio(keep_channels=k),
    lambda k: F.Peak(keep_channels=k),
    lambda k: F.Peak(db=True, keep_channels=k),
    lambda k: F.DCOffset(keep_channels=k),
    lambda k: F.DynamicRange(frame_length=512, keep_channels=k),
    lambda k: F.SilenceRatio(frame_length=512, keep_channels=k),
    lambda k: F.OnsetStrength(keep_channels=k),
    lambda k: F.OnsetRate(keep_channels=k),
    lambda k: F.Tempo(keep_channels=k),
    lambda k: F.Chroma(keep_channels=k),
    lambda k: F.Chroma(norm=None, keep_channels=k),
    lambda k: F.Harmonicity(keep_channels=k),
    lambda k: F.Inharmonicity(keep_channels=k),
    lambda k: F.Gammatone(keep_channels=k),
    lambda k: F.Gammatone(db=True, n_bands=32, keep_channels=k),
    lambda k: F.Cepstrum(keep_channels=k),
    lambda k: F.Cepstrum(n_coeffs=64, window=None, keep_channels=k),
    lambda k: F.Loudness(keep_channels=k),
    lambda k: F.Loudness(frame_length=4096, mode="db", keep_channels=k),
]


@pytest.mark.parametrize("keep_channels", [False, True])
@pytest.mark.parametrize("make", FEATURES, ids=[str(i) for i in range(len(FEATURES))])
def test_feature_nn_module(make, keep_channels):
    feature = make(keep_channels)
    x = _signal(2)
    expected = feature(x.clone())
    module, scripted = _modules(feature)
    for m in (module, scripted):
        _close(m(x.clone()), expected)


# (feature, input channels, invert kwargs of the feature)
INVERTIBLE_FEATURES = [
    (lambda k: F.DCT(hop_length=1024, keep_channels=k), 2, {"length": N_SAMPLES}),
    (lambda k: F.DCT(n_coeffs=512, window=None, keep_channels=k), 2, {"length": N_SAMPLES}),
    (lambda k: F.STFT(mode="complex", keep_channels=k), 2, {}),
    (lambda k: F.STFT(mode=["magnitude", "phase"], keep_channels=k), 2, {}),
]


@pytest.mark.parametrize("keep_channels", [False, True])
@pytest.mark.parametrize("make, n_channels, invert_kwargs", INVERTIBLE_FEATURES,
                         ids=[str(i) for i in range(len(INVERTIBLE_FEATURES))])
def test_invertible_feature_nn_module(make, n_channels, invert_kwargs, keep_channels):
    feature = make(keep_channels)
    x = _signal(n_channels)
    expected = feature(x.clone())
    expected_inverse = feature.invert(expected.clone(), **invert_kwargs)
    module, scripted = _modules(feature)
    for m in (module, scripted):
        _close(m(x.clone()), expected)
        inverse = m.invert(expected.clone(), *invert_kwargs.values())
        _close(inverse, expected_inverse)


@pytest.mark.parametrize("inverse", [False, True])
def test_pqmf_nn_module(inverse):
    feature = F.PQMF(n_bands=4, inverse=inverse)
    x = _signal(2) if not inverse else _signal(8, n_samples=N_SAMPLES // 4)
    expected = feature(x.clone())
    module, scripted = _modules(feature)
    for m in (module, scripted):
        _close(m(x.clone()), expected)
        if inverse:
            _close(m.invert(expected.clone()), feature.invert(expected.clone()))
        else:
            _close(m.invert(expected.clone(), N_SAMPLES), feature.invert(expected.clone(), length=N_SAMPLES))


def test_mel_nn_module():
    feature = F.Mel(sr=SR, n_fft=1024, n_mels=64)
    x = _signal(2)
    module, scripted = _modules(feature)
    for m in (module, scripted):
        _close(m(x.clone()), feature(x.clone()))


def test_feature_without_nn_module():
    with pytest.raises(NotImplementedError):
        F.LoudnessMeter().nn_module()


def test_scripted_module_save_load(tmp_path):
    """scripted modules can be saved and reloaded (e.g. embedded in exported models)."""
    module = torch.jit.script(T.MidSide().nn_module())
    torch.jit.save(module, str(tmp_path / "midside.ts"))
    loaded = torch.jit.load(str(tmp_path / "midside.ts"))
    x = _signal(2)
    _close(loaded.invert(loaded(x)), x, atol=1e-6)


# --------------------------------------------------------------------------------------------------------------------- #
#  probability guard
# --------------------------------------------------------------------------------------------------------------------- #

DETERMINISTIC_TRANSFORMS = [
    lambda **kw: T.MidSide(**kw),
    lambda **kw: T.AmbisonicEncoder(**kw),
    lambda **kw: T.AmbisonicDecoder(**kw),
    lambda **kw: MiscResample(sr=SR, target_sr=22050, **kw),
    lambda **kw: T.PreEmphasis(**kw),
    lambda **kw: T.Log1p(**kw),
    lambda **kw: T.Exp1m(**kw),
    lambda **kw: MiscNormalize(**kw),
    lambda **kw: T.Derivator(**kw),
    lambda **kw: T.Integrator(sr=SR, **kw),
    lambda **kw: T.CompressRange(**kw),
    lambda **kw: T.NormRange(**kw),
]


@pytest.mark.parametrize("make", DETERMINISTIC_TRANSFORMS, ids=[str(i) for i in range(len(DETERMINISTIC_TRANSFORMS))])
def test_deterministic_transform_refuses_p(make):
    transform = make()
    assert transform.p is None and not transform.allow_random
    with pytest.raises(T.AcidsTransformException):
        make(p=0.5)


@pytest.mark.parametrize("transform", [MiscGain(p=0.5), T.Mute(p=0.3), T.Stretch(sr=SR, p=0.5)])
def test_random_transform_accepts_p(transform):
    assert transform.allow_random and transform.p is not None
