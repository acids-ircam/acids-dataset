"""Spatial features on reference signals."""
import pytest
import torch
import numpy as np, scipy.signal
from acids_dataset.transforms import MidSide
from acids_dataset.features import Width, SpatialCovariance, CrossSpectralMatrix, SpectroSpatialCovarianceVector, GeneralizedCrossCorrelation

N = 22050


@pytest.fixture
def signals():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(N, generator=g), torch.randn(N, generator=g)
    return {"identical": torch.stack([a, a]), "opposite": torch.stack([a, -a]), "uncorrelated": torch.stack([a, b]), 
            "mono": a[None], "quad": torch.stack([a, a, a, a]), "silence": torch.zeros(2, N)}


def test_width_width(signals):
    expected = {"identical": 0., "opposite": 1., "uncorrelated": 0.5, "mono": 0., "quad": 0., "silence": 0.}
    for name, x in signals.items():
        width = Width(frame_length=2048)(x)
        assert width.shape == (N // 2048,)
        assert abs(float(width.mean()) - expected[name]) < 0.05, name
        assert torch.all((width >= 0) & (width <= 1))


def test_width_modes(signals):
    ratio = Width(mode="ratio")(signals["uncorrelated"])
    assert torch.allclose(ratio, torch.zeros_like(ratio), atol=0.5)
    energy = Width(mode="energy")(signals["identical"])
    assert energy.shape == (2, N // 2048) and torch.all(energy[1] < -90)
    # stereo definition : mid = (L + R) / 2, side = (L - R) / 2
    x = signals["uncorrelated"]
    mid_db = 10 * torch.log10(((x[0] + x[1]) / 2)[:2048].square().mean())
    assert abs(float(energy[0, 0]) - float(10 * torch.log10(((signals["identical"][0]) ** 2)[:2048].mean()))) < 1e-3
    assert abs(float(Width(mode="energy")(x)[0, 0]) - float(mid_db)) < 1e-3
    with pytest.raises(AssertionError):
        Width(mode="unknown")


def test_width_framing():
    assert Width(frame_length=1024, hop_length=512)(torch.randn(2, 4096)).shape == (7,)
    assert Width(frame_length=2048)(torch.randn(2, 100)).shape == (1,)
    assert Width()(torch.randn(3, 2, 4096)).shape == (3, 2)


def test_midside_transform(signals):
    x = signals["uncorrelated"]
    ms = MidSide()(x)
    assert ms.shape == (2, N)
    assert torch.allclose(ms[0], (x[0] + x[1]) / 2) and torch.allclose(ms[1], (x[0] - x[1]) / 2)
    assert torch.allclose(MidSide(inverse=True)(ms), x, atol=1e-6)
    # orthonormal : energy preserving, and its own inverse
    ortho = MidSide(normalization="orthonormal")
    assert torch.allclose(ortho(x).square().sum(), x.square().sum(), rtol=1e-4)
    assert torch.allclose(ortho(ortho(x)), x, atol=1e-5)
    # side of identical channels is zero ; batches
    assert torch.all(MidSide()(signals["identical"])[1] == 0)
    assert MidSide()(torch.randn(3, 2, 100)).shape == (3, 2, 100)
    assert MidSide().name == "midside" and MidSide(inverse=True).name == "leftright"
    # numpy inputs give numpy outputs
    assert isinstance(MidSide()(np.random.randn(2, 100)), np.ndarray)


@pytest.mark.parametrize("name", ["mono", "quad"])
def test_midside_needs_stereo(signals, name):
    with pytest.raises(ValueError):
        MidSide()(signals[name])


def test_spatial_covariance(signals):
    cov = SpatialCovariance(frame_length=2048)(signals["opposite"])
    assert cov.shape == (2, 2, N // 2048)
    assert torch.allclose(cov[..., 0], torch.tensor([[1., -1.], [-1., 1.]]), atol=1e-4)
    uncorrelated = SpatialCovariance()(signals["uncorrelated"])
    assert float(uncorrelated[0, 1].abs().mean()) < 0.1
    raw = SpatialCovariance(normalize=False)(signals["identical"])
    assert torch.allclose(raw[0, 0], signals["identical"][0, :raw.shape[-1] * 2048].reshape(-1, 2048).square().mean(-1), atol=1e-4)
    assert torch.all(SpatialCovariance()(signals["silence"]) == 0)


def test_spatial_covariance_bands(signals):
    cov = SpatialCovariance(frame_length=2048, n_bands=4)(signals["identical"])
    assert cov.shape == (2, 4, 2, 2, N // 2048)
    # identical channels : coherence of 1, without imaginary part
    assert torch.allclose(cov[0], torch.ones_like(cov[0]), atol=1e-4) and torch.allclose(cov[1], torch.zeros_like(cov[1]), atol=1e-4)
    coherence = SpatialCovariance(n_bands=4)(signals["uncorrelated"])
    assert float(torch.sqrt(coherence[0, :, 0, 1] ** 2 + coherence[1, :, 0, 1] ** 2).mean()) < 0.15
    assert SpatialCovariance(n_bands=2)(torch.randn(3, 4, 100)).shape == (3, 2, 2, 4, 4, 1)


def test_cross_spectral_matrix_scipy(signals):
    # C[i, j] = csd(x_j, x_i) on each frame
    x = signals["uncorrelated"] + 0.5 * torch.roll(signals["uncorrelated"].flip(0), 3, -1)
    sr = 22050
    csm = CrossSpectralMatrix(frame_length=4096, n_fft=512, sr=sr)(x)
    assert csm.shape == (2, 257, 2, 2, N // 4096)
    frame = x[:, 4096:8192].double().numpy()
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        _, reference = scipy.signal.csd(frame[j], frame[i], fs=sr, nperseg=512, detrend=False)
        assert np.allclose(csm[0, :, i, j, 1].numpy(), reference.real, atol=1e-6, rtol=1e-3), (i, j)
        assert np.allclose(csm[1, :, i, j, 1].numpy(), reference.imag, atol=1e-6, rtol=1e-3), (i, j)
    # hermitian
    assert torch.allclose(csm[0], csm[0].transpose(1, 2), atol=1e-6) and torch.allclose(csm[1], -csm[1].transpose(1, 2), atol=1e-6)


def test_cross_spectral_matrix_modes(signals):
    coherence = lambda x: CrossSpectralMatrix(frame_length=8192, n_fft=512, mode="coherence")(x)
    assert torch.allclose(coherence(signals["identical"]), torch.ones(1), atol=1e-4)
    assert float(coherence(signals["uncorrelated"])[:, 0, 1].mean()) < 0.15
    assert float(coherence(torch.stack([signals["identical"][0], torch.roll(signals["identical"][0], 5)]))[:, 0, 1].mean()) > 0.95
    assert torch.all(coherence(signals["silence"]) == 0)
    # a delay of d samples gives a phase of 2 pi f d / sr
    a = signals["identical"][0]
    phase = CrossSpectralMatrix(frame_length=8192, n_fft=1024, mode="phase", sr=22050)(torch.stack([a, torch.roll(a, 2)]))
    f = torch.from_numpy(CrossSpectralMatrix(n_fft=1024, sr=22050).frequencies).float()
    low = f < 2000
    assert torch.allclose(phase[low, 0, 1, 0], 2 * np.pi * f[low] * 2 / 22050, atol=0.05)
    magnitude = CrossSpectralMatrix(mode="magnitude")(signals["opposite"])
    assert magnitude.shape == (513, 2, 2, N // 4096) and torch.allclose(magnitude[:, 0, 1], magnitude[:, 0, 0], rtol=1e-4)
    with pytest.raises(AssertionError):
        CrossSpectralMatrix(mode="unknown")


def test_cross_spectral_matrix_bins():
    csm = CrossSpectralMatrix(n_fft=1024, fmin=100., fmax=4000., sr=22050)
    assert csm.frequencies.min() >= 100 and csm.frequencies.max() <= 4000
    assert csm(torch.randn(2, N)).shape == (2, len(csm.frequencies), 2, 2, N // 4096)
    banded = CrossSpectralMatrix(n_fft=1024, n_bands=8)
    assert len(banded.frequencies) == 8 and banded(torch.randn(4, N)).shape == (2, 8, 4, 4, N // 4096)
    # batches, short and mono inputs
    x = torch.randn(3, 2, N)
    assert torch.allclose(banded.batch_call(x), torch.stack([banded(x[i]) for i in range(3)]), atol=1e-5)
    assert CrossSpectralMatrix(n_fft=256)(torch.randn(2, 100)).shape == (2, 129, 2, 2, 1)
    assert CrossSpectralMatrix(n_fft=256, mode="coherence")(torch.randn(N)).shape == (129, 1, 1, N // 4096)


def _reference_sscv(x, sr, frame_length, hop_length, n_bands, frame, band):
    """SSCV of one frame and band, following equations (1-6) of arXiv:2411.03172."""
    import torchaudio
    x = x.double().numpy()
    segment = x[:, frame * hop_length:frame * hop_length + frame_length] * np.hanning(frame_length + 1)[:-1]
    spectrum = np.fft.rfft(segment, axis=-1)                                          # (M, K)
    w = torchaudio.functional.melscale_fbanks(frame_length // 2 + 1, 0., sr / 2, n_bands, sr)[:, band].double().numpy()
    w = w / w.sum()
    cov = sum(w[k] * np.outer(spectrum[:, k], spectrum[:, k].conj()) for k in range(len(w))) / (w > 0).sum()
    m = cov.shape[0]
    c = list(np.diag(cov))
    for i in range(m):
        for j in range(i + 1, m):
            c += [cov[i, j], cov[j, i]]
    c = np.array(c)
    r = np.zeros(m * m)
    r[0] = c[:m].real.sum() / np.sqrt(m)                                               # first row of the real DFT
    rotation = np.array([[1, 1], [-1j, 1j]]) / np.sqrt(2)
    for k in range(m, m * m, 2):
        r[k:k + 2] = (rotation @ c[k:k + 2]).real
    return np.log(r[0]), r[m:] / r[0], cov


def test_sscv_reference():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(N, generator=g)
    foa = torch.stack([a, 0.5 * torch.roll(a, 3), torch.randn(N, generator=g), 0.2 * a + torch.randn(N, generator=g)])
    feature = SpectroSpatialCovarianceVector(sr=16000)
    assert (feature.frame_length, feature.hop_length) == (1536, 768)
    sscv = feature(foa)
    assert sscv.shape == (16, 52, 1 + (N - 1536) // 768)
    for frame, band in [(0, 5), (3, 20), (10, 51)]:
        log_energy, pairs, cov = _reference_sscv(foa, 16000, 1536, 768, 52, frame, band)
        assert abs(float(sscv[0, band, frame]) - log_energy) < 1e-3
        assert np.allclose(sscv[4:, band, frame].numpy(), pairs, atol=1e-4)
        # diagonal : orthonormal real DFT (energy preserved)
        r0 = np.exp(log_energy)
        assert np.allclose(np.sum((sscv[:4, band, frame].numpy()[1:] * r0) ** 2) + r0 ** 2, np.sum(np.diag(cov).real ** 2), rtol=1e-3)


def test_sscv_properties(signals):
    feature = SpectroSpatialCovarianceVector(sr=22050)
    quad = feature(signals["quad"])
    # identical channels : flat diagonal, real off-diagonal pairs equal to the energy
    assert torch.allclose(quad[1:4], torch.zeros(1), atol=1e-5)
    assert torch.allclose(quad[4::2], torch.full((1,), 2 ** -0.5), atol=1e-5) and torch.allclose(quad[5::2], torch.zeros(1), atol=1e-5)
    # gain : only the log energy changes
    x = torch.randn(4, N)
    reference, louder = feature(x), feature(3 * x)
    assert torch.allclose(louder[0] - reference[0], torch.full((1,), 2 * np.log(3.)), atol=1e-4)
    assert torch.allclose(louder[1:], reference[1:], atol=1e-4)
    # delays give imaginary parts
    a = signals["identical"][0]
    assert float(feature(torch.stack([a, torch.roll(a, 4)]))[3].abs().mean()) > 0.1
    # silence, mono, short inputs and batches
    silence = feature(torch.zeros(4, N))
    assert torch.all(torch.isfinite(silence)) and torch.all(silence[1:] == 0)
    assert feature(signals["mono"]).shape[0] == 1
    assert SpectroSpatialCovarianceVector(sr=22050, n_bands=16)(torch.randn(3, 4, 100)).shape == (3, 16, 16, 1)
    with pytest.raises(AssertionError):
        SpectroSpatialCovarianceVector(alpha=1.)


def test_sscv_smoothing():
    x = torch.randn(4, N)
    raw, smoothed = SpectroSpatialCovarianceVector(sr=22050), SpectroSpatialCovarianceVector(sr=22050, alpha=0.8)
    assert torch.equal(raw(x), SpectroSpatialCovarianceVector(sr=22050, alpha=0.)(x))
    cov, cov_smoothed = raw.covariance(x), smoothed.covariance(x)
    assert torch.allclose(cov_smoothed[0], cov[0])
    assert torch.allclose(cov_smoothed[2], 0.2 * cov[2] + 0.8 * (0.2 * cov[1] + 0.8 * cov[0]), rtol=1e-4, atol=1e-6)
    # stationary noise : smoothing reduces the variance across frames
    assert float(smoothed(x)[4:].std(-1).mean()) < float(raw(x)[4:].std(-1).mean())


def fractional_delay(x, delay):
    f = torch.fft.rfftfreq(x.shape[-1])
    return torch.fft.irfft(torch.fft.rfft(x) * torch.exp(-2j * np.pi * f * delay), n=x.shape[-1])


@pytest.mark.parametrize("weighting", GeneralizedCrossCorrelation.weightings)
def test_gcc_delays(signals, weighting):
    a = signals["uncorrelated"][0]
    for delay in [3, 7, -5]:
        x = torch.stack([a, torch.roll(a, delay)])
        for kwargs in [{}, {"frame_length": 4096, "n_fft": 512}]:
            gcc = GeneralizedCrossCorrelation(weighting=weighting, sr=44100, **kwargs)
            assert gcc(x).mean().item() == pytest.approx(delay, abs=1e-3)
    seconds = GeneralizedCrossCorrelation(weighting=weighting, unit="seconds", sr=44100)(torch.stack([a, torch.roll(a, 7)]))
    assert seconds.mean().item() == pytest.approx(7 / 44100, abs=1e-7)


def test_gcc_interpolation(signals):
    a = signals["uncorrelated"][0]
    x = torch.stack([a, fractional_delay(a, 3.4)])
    coarse = GeneralizedCrossCorrelation(sr=44100)(x).mean().item()
    fine = GeneralizedCrossCorrelation(interp=8, sr=44100)(x).mean().item()
    assert abs(fine - 3.4) < 0.02 and abs(fine - 3.4) < abs(coarse - 3.4)


def test_gcc_correlation(signals):
    a, b = signals["uncorrelated"]
    gcc = GeneralizedCrossCorrelation(mode="correlation", interp=2, sr=44100)
    out = gcc(torch.stack([a, torch.roll(a, 7)]))
    assert out.shape == (1, len(gcc.lags), N // 2048) and len(gcc.lags) == 2 * round(1e-3 * 44100 * 2) + 1
    assert gcc.lags[out[0].argmax(0)].tolist() == [7.] * out.shape[-1]
    assert torch.allclose(out.amax(-2), torch.ones(1), atol=0.01)
    plain = GeneralizedCrossCorrelation(mode="correlation", weighting="none", sr=44100)
    assert plain(torch.stack([a, -a])).amin(-2).mean().item() == pytest.approx(-1, abs=1e-4)
    assert plain(torch.stack([a, b])).abs().amax(-2).mean().item() < 0.15
    assert GeneralizedCrossCorrelation(max_delay=None, n_fft=256, mode="correlation", sr=44100).lags.tolist() == list(range(-255, 256))


def test_gcc_pairs(signals):
    a = signals["uncorrelated"][0]
    x = torch.stack([a, torch.roll(a, 3), torch.roll(a, 8)])
    assert GeneralizedCrossCorrelation(sr=44100)(x).mean(-1).tolist() == pytest.approx([3, 8], abs=1e-3)
    assert GeneralizedCrossCorrelation(pairs="all", sr=44100)(x).mean(-1).tolist() == pytest.approx([3, 8, 5], abs=1e-3)
    # mono and silent inputs give zero delays
    assert torch.all(GeneralizedCrossCorrelation(sr=44100)(signals["mono"]) == 0)
    assert torch.all(GeneralizedCrossCorrelation(sr=44100)(signals["silence"]) == 0)
