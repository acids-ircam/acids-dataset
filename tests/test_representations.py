"""STFT, FFT, PQMF, DCT, gammatone, cepstrum and LPC features on reference signals."""
import math
import numpy as np
import pytest
import scipy.fft, scipy.linalg, scipy.signal
import torch
from acids_dataset.features import STFT, FFT, PQMF, DCT, Gammatone, Cepstrum, LPC

SR = 16000


@pytest.fixture
def noise():
    return torch.randn(1, SR, generator=torch.Generator().manual_seed(0))


@pytest.fixture
def ar_process():
    a = np.array([1, -2.2137, 2.9403, -2.1697, 0.9606])
    e = np.random.default_rng(0).standard_normal(2 * SR)
    return a, torch.from_numpy(scipy.signal.lfilter([1], a, e).astype(np.float32))[None]


def test_dct(noise):
    out = DCT(frame_length=1024, window=None)(noise)
    assert out.shape == (1024, SR // 1024)
    assert np.allclose(out[:, 0].numpy(), scipy.fft.dct(noise[0, :1024].numpy(), norm="ortho"), atol=1e-4)
    assert DCT(frame_length=512, n_coeffs=20, keep_channels=True)(torch.randn(3, 2, 4096)).shape == (3, 2, 20, 8)


def test_cepstrum(noise):
    out = Cepstrum(frame_length=1024, window=None)(noise)
    reference = np.fft.irfft(np.log(np.abs(np.fft.rfft(noise[0, :1024].numpy())) + 1e-10))
    assert out.shape == (513, SR // 1024)
    assert np.allclose(out[:, 0].numpy(), reference[:513], atol=1e-4)
    # harmonic signals have a cepstral peak at sr / f0
    t = torch.arange(SR) / SR
    harmonic = sum(torch.sin(2 * math.pi * 200 * k * t) / k for k in range(1, 20))[None]
    cepstrum = Cepstrum(frame_length=2048, sr=SR)(harmonic).mean(-1)
    assert int(cepstrum[40:400].argmax()) + 40 == SR // 200


def test_gammatone():
    feature = Gammatone(frame_length=1024, n_bands=32, fmin=50, sr=SR)
    assert feature.feature_name == "gammatone_32"
    assert abs(float(feature.center_frequencies[0]) - 50) < 1e-6 and abs(float(feature.center_frequencies[-1]) - SR / 2) < 1e-3
    t = torch.arange(SR) / SR
    for f in [200., 1000., 4000.]:
        bands = feature(torch.sin(2 * math.pi * f * t)[None]).mean(-1)
        closest = int((feature.center_frequencies - f).abs().argmin())
        assert int(bands.argmax()) == closest
    assert Gammatone(db=True, sr=SR, keep_channels=True)(torch.randn(4, 2, SR)).shape == (4, 2, 64, SR // 2048)


def test_lpc(ar_process):
    a_true, x = ar_process
    coefficients = LPC(frame_length=4096, order=4, window=None, sr=SR)(x)
    assert coefficients.shape == (4, x.shape[-1] // 4096)
    frame = x[0, :4096].double().numpy()
    r = np.correlate(frame, frame, "full")[4095:4100]
    assert np.allclose(coefficients[:, 0].numpy(), scipy.linalg.solve_toeplitz(r[:4], -r[1:5]), atol=1e-3)
    assert np.allclose(coefficients.mean(-1).numpy(), a_true[1:], atol=0.1)


@pytest.mark.parametrize("order", [4, 5, 16])
def test_lsf(ar_process, order):
    _, x = ar_process
    lsf = LPC(order=order, mode="lsf", sr=SR)(x)
    assert lsf.shape[0] == order
    assert torch.all(lsf.diff(dim=0) > 0) and torch.all((lsf > 0) & (lsf < math.pi))
    a = LPC(order=order, sr=SR)(x)[:, 0].double().numpy()
    extended = np.r_[1, a, 0]
    roots = np.r_[np.roots(extended + extended[::-1]), np.roots(extended - extended[::-1])]
    angles = np.sort(np.abs(np.angle(roots)))
    angles = angles[(angles > 1e-6) & (angles < math.pi - 1e-6)][::2]
    assert np.allclose(lsf[:, 0].numpy(), angles, atol=1e-4)


def test_lpc_modes(ar_process):
    _, x = ar_process
    reflection = LPC(order=8, mode="reflection", sr=SR)(x)
    assert torch.all(reflection.abs() < 1)
    envelope = LPC(order=16, mode="envelope", frame_length=2048, sr=SR)(x)
    assert envelope.shape == (1025, x.shape[-1] // 2048)
    spectrum = 10 * torch.log10(torch.fft.rfft(x[0, :2048] * torch.hann_window(2048)).abs() ** 2 + 1e-10)
    assert abs(float((envelope[:, 0] - spectrum).median())) < 3
    assert LPC(order=4, gain=True, sr=SR)(x).shape[0] == 5
    assert LPC(mode="lsf").feature_name == "lpc_lsf" and LPC().feature_name == "lpc"
    # silent frames : A(z) = 1
    assert torch.all(LPC(order=4)(torch.zeros(1, 4096)) == 0)
    assert LPC(keep_channels=True, pre_emphasis=0.97)(torch.randn(3, 2, 8192)).shape == (3, 2, 16, 4)


def test_stft(noise):
    out = STFT(frame_length=1024, hop_length=256, mode="complex", center=True, n_fft=2048, sr=SR)(noise)
    padded = torch.nn.functional.pad(noise[0], (512, 512))
    reference = torch.fft.rfft(padded.unfold(-1, 1024, 256) * torch.hann_window(1024), n=2048).T
    assert out.shape == (1025, 1 + SR // 256) and out.is_complex()
    assert torch.allclose(out, reference, atol=1e-3)
    # power spectral density, as scipy
    psd = STFT(frame_length=512, hop_length=128, mode="power_density", sr=SR)(noise)
    _, _, reference = scipy.signal.spectrogram(noise[0].numpy(), fs=SR, window="hann", nperseg=512, noverlap=384, detrend=False,
                                               scaling="density", mode="psd")
    assert np.allclose(psd.numpy(), reference, rtol=1e-3, atol=1e-10)
    psd = STFT(frame_length=500, hop_length=500, n_fft=1024, window=None, mode="power_density", sr=SR)(noise)
    _, _, reference = scipy.signal.spectrogram(noise[0].numpy(), fs=SR, window="boxcar", nperseg=500, noverlap=0, nfft=1024,
                                               detrend=False, scaling="density", mode="psd")
    assert np.allclose(psd.numpy(), reference, rtol=1e-3, atol=1e-10)


def test_stft_modes():
    x = torch.randn(3, 2, SR)
    out = STFT(mode=["magnitude", "phase", "power", "db", "real", "imaginary"], keep_channels=True, sr=SR)(x)
    assert out.shape == (3, 12, 1025, 28) and not out.is_complex()
    magnitude, phase = out[:, :2], out[:, 2:4]
    assert torch.allclose(torch.polar(magnitude, phase), torch.complex(out[:, 8:10], out[:, 10:12]), atol=1e-3)
    assert torch.allclose(out[:, 4:6], magnitude.square(), rtol=1e-4, atol=1e-6)
    # complex outputs only with the complex mode
    out = STFT(mode=["magnitude", "complex"], sr=SR)(x)
    assert out.shape == (3, 2, 1025, 28) and out.is_complex() and torch.all(out[:, 0].imag == 0)
    assert STFT(sr=SR)(x).shape == (3, 1025, 28)
    assert STFT().feature_name == "stft" and STFT(mode=["real", "imaginary"]).feature_name == "stft_real_imaginary"
    with pytest.raises(AssertionError):
        STFT(mode="unknown")


def test_instantaneous_frequency():
    t = torch.arange(SR) / SR
    sine = torch.sin(2 * math.pi * 1003.7 * t)[None]
    magnitude, frequency, phase = STFT(frame_length=1024, hop_length=256, mode=["magnitude", "instantaneous_frequency", "instantaneous_phase"], sr=SR)(sine)
    peak = int(magnitude.mean(-1).argmax())
    assert torch.allclose(frequency[peak - 1:peak + 2, 1:], torch.tensor(1003.7), atol=0.1)
    assert torch.all(phase[:, 0] == 0) and torch.all(phase.abs() <= math.pi + 1e-5)


def test_fft(noise):
    assert torch.allclose(FFT(mode="complex", sr=SR)(noise), torch.fft.rfft(noise[0]), atol=1e-2)
    assert FFT(onesided=False, sr=SR)(noise).shape == (SR,)
    assert torch.allclose(FFT(onesided=False, sr=SR).frequencies(8), torch.fft.fftfreq(8, 1 / SR))
    for kwargs, scipy_kwargs in [({"window": "hann"}, {"window": "hann"}), ({"n_fft": 32768}, {"nfft": 32768}),
                                 ({"onesided": False}, {"return_onesided": False})]:
        _, reference = scipy.signal.periodogram(noise[0].numpy(), fs=SR, detrend=False, scaling="density", **scipy_kwargs)
        assert np.allclose(FFT(mode="power_density", sr=SR, **kwargs)(noise).numpy(), reference, rtol=1e-3, atol=1e-12)
    assert FFT(mode=["db", "phase"], keep_channels=True, sr=SR)(torch.randn(4, 2, 1000)).shape == (4, 4, 501)
    assert FFT(mode=["db", "phase"])(torch.randn(2, 1000)).shape == (2, 501)
    assert FFT(n_fft=500)(torch.randn(2, 1000)).shape == (251,)
    with pytest.raises(AssertionError):
        FFT(mode="instantaneous_phase")


@pytest.mark.parametrize("n_bands", [2, 4, 16])
def test_pqmf(n_bands):
    x = torch.randn(2, 3, SR, generator=torch.Generator().manual_seed(0))
    bands = PQMF(n_bands=n_bands, sr=SR)(x)
    assert bands.shape == (2, 3 * n_bands, math.ceil(SR / n_bands))
    reconstruction = PQMF(n_bands=n_bands, inverse=True, sr=SR)(bands)[..., :SR]
    error = (reconstruction - x)[..., 2000:-2000].square().mean() / x[..., 2000:-2000].square().mean()
    assert 10 * torch.log10(error) < -50
    with pytest.raises(ValueError):
        PQMF(n_bands=4, inverse=True)(torch.randn(3, 100))


def test_pqmf_bands():
    pqmf = PQMF(n_bands=8, sr=SR)
    t = torch.arange(SR) / SR
    for f in [300., 2500., 7000.]:
        energies = pqmf(torch.sin(2 * math.pi * f * t)[None]).square().mean(-1)
        assert int(energies.argmax()) == int(f // (SR / 2 / 8))
    assert pqmf.feature_name == "pqmf_8" and PQMF(n_bands=8, inverse=True).feature_name == "pqmf_8_inverse"


@pytest.mark.parametrize("kwargs", [dict(mode="complex"), dict(mode=["magnitude", "phase"], keep_channels=True),
                                    dict(mode=["real", "imaginary"]), dict(mode=["db", "phase"], center=True),
                                    dict(mode=["power_density", "phase"], n_fft=2048, keep_channels=True)])
def test_stft_inversion(kwargs):
    """STFT features are inverted exactly from complex outputs, or magnitudes and phases (acids_transforms)."""
    x = torch.randn(3, 2, SR, generator=torch.Generator().manual_seed(0))
    feature = STFT(frame_length=1024, hop_length=256, sr=SR, **kwargs)
    y = feature.invert(feature(x))
    target = x if feature.keep_channels else x.mean(-2)
    assert y.shape[:-1] == target.shape[:-1]
    # borders without center are covered by vanishing windows
    lo = 0 if feature.center else 1024
    n = min(y.shape[-1], target.shape[-1]) - lo
    assert torch.allclose(y[..., lo:n], target[..., lo:n], atol=1e-4)


def test_stft_magnitude_inversion(noise):
    feature = STFT(frame_length=512, hop_length=128, sr=SR)
    magnitude = feature(noise[:, :8192])
    for inversion_mode in [None, "random", "keep_input"]:
        y = feature.invert(magnitude, inversion_mode=inversion_mode)
        assert y.ndim == 1 and abs(y.shape[-1] - 8192) <= 512
    # griffin-lim converges towards the magnitude
    y = feature.invert(magnitude)
    assert (feature(y[:8192]) - magnitude).norm() / magnitude.norm() < 0.3


@pytest.mark.parametrize("mode", ["magnitude", ["magnitude", "phase"]])
def test_stft_script(mode):
    x = torch.randn(3, 2, SR)
    feature = STFT(frame_length=1024, hop_length=256, mode=mode, keep_channels=True, sr=SR)
    scripted = torch.jit.script(feature.script())
    assert torch.allclose(scripted(x), feature(x), atol=1e-5)
    if mode != "magnitude":
        assert torch.allclose(scripted.invert(scripted(x)), feature.invert(feature(x)), atol=1e-5)


@pytest.mark.parametrize("kwargs", [dict(), dict(mode=["magnitude", "instantaneous_frequency"], keep_channels=True),
                                    dict(mode="complex", n_fft=2048)])
def test_stft_streaming(kwargs):
    """streaming STFT features are the ones of the feature (center=False), and are inverted chunk by chunk."""
    x = torch.randn(2, 2, 64 * 256, generator=torch.Generator().manual_seed(0))
    feature = STFT(frame_length=1024, hop_length=256, sr=SR, **kwargs)
    reference = feature(x)
    for streaming in [feature.streaming(), torch.jit.script(feature.streaming())]:
        frames = torch.cat([streaming(c) for c in x.split(2048, -1)], -1)
        k = 1024 // 256 - 1
        assert frames.shape[-1] == x.shape[-1] // 256
        # (instantaneous modes have no previous frame for the first frame of the feature)
        assert torch.allclose(frames[..., k + 1:k + reference.shape[-1]], reference[..., 1:], atol=1e-4)
        if feature.mode == ["complex"]:
            streaming.reset()
            y = torch.cat([streaming.invert(streaming(c)) for c in x.split(2048, -1)], -1)
            latency = streaming.latency
            assert latency == 768 and torch.allclose(y[..., latency:], x.mean(-2)[..., :-latency], atol=1e-4)


def test_stft_pickling():
    import pickle
    feature = STFT(frame_length=512, sr=SR)
    x = torch.randn(1, 4096)
    out = feature(x)
    restored = pickle.loads(pickle.dumps(feature))
    assert torch.allclose(restored(x), out)
    # features pickled without acids_transforms modules
    del restored.__dict__["_module"]
    assert torch.allclose(restored(x), out)
