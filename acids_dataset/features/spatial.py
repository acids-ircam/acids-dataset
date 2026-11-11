"""Spatial audio features : stereo width (generalized to any number of channels), spatial
covariance matrices and inter-channel descriptors."""
import gin
import numpy as np
import torch
from typing import Optional

from . import RAISE_EXC_IF_FEATURE_ERROR
from .base import AcidsDatasetFeature

from .framed import _EPS, _frames, _as_tensor, _FramedFeature, _WaveformFeature


@gin.configurable(module="ad.features")
class Width(_FramedFeature):
    """Stereo width per frame, from a mid / side analysis. The mid signal is the mean of channels, and the side signal
    the residual of each channel (x_c - mid) : for stereo signals, mid = (L + R) / 2 and side = +/- (L - R) / 2, and the
    generalization holds for any number of channels (side energy is averaged over channels).

    Modes :
        - "width" (default) : side energy / (mid energy + side energy), in [0, 1] (0 for identical channels, 0.5 for
          uncorrelated channels of equal power, 1 for channels summing to zero). Shape (..., n_frames).
        - "ratio" : side to mid energy ratio, in dB. Shape (..., n_frames).
        - "energy" : mid and side energies, in dB. Shape (..., 2, n_frames).
    Frames are non-overlapping by default (hop_length = frame_length) ; silent frames have a width of 0.
    """
    modes = ["width", "ratio", "energy"]

    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, mode: str = "width", sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, sr=sr, **kwargs)
        assert mode in self.modes, "mode must be in %s, got %s"%(self.modes, mode)
        self.mode = mode

    def __repr__(self):
        return "Width(mode=%s, frame_length=%d, hop_length=%d)"%(self.mode, self.frame_length, self.hop_length)

    @property
    def default_feature_name(self):
        return "width"

    def __call__(self, data) -> torch.Tensor:
        x = _as_tensor(data)
        mid = x.mean(dim=-2, keepdim=True)
        side = x - mid
        mid_energy = torch.square(_frames(mid, self.frame_length, self.hop_length)).mean(-1)[..., 0, :]
        side_energy = torch.square(_frames(side, self.frame_length, self.hop_length)).mean(-1).mean(-2)
        if self.mode == "width":
            total = mid_energy + side_energy
            return torch.where(total > _EPS, side_energy / total.clamp(min=_EPS), torch.zeros_like(total))
        mid_db, side_db = 10 * torch.log10(mid_energy + _EPS), 10 * torch.log10(side_energy + _EPS)
        if self.mode == "ratio":
            return side_db - mid_db
        return torch.stack([mid_db, side_db], dim=-2)


@gin.configurable(module="ad.features")
class SpatialCovariance(_FramedFeature):
    """Spatial covariance matrices between channels, per frame.

    If n_bands is None, covariances are computed on the waveform : shape (..., C, C, n_frames). Otherwise, frames are
    analyzed with a hann-windowed FFT (n_fft = frame_length), and complex spatial covariance matrices are summed over
    n_bands linear frequency bands : the output stacks their real and imaginary parts, with shape
    (..., 2, n_bands, C, C, n_frames). If normalize, matrices are normalized by channel energies (correlation /
    coherence matrices, with ones on the diagonal for non-silent channels).
    """

    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, n_bands: int | None = None, normalize: bool = True,
                 sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, sr=sr, **kwargs)
        self.n_bands = n_bands
        self.normalize = normalize

    def __repr__(self):
        return "SpatialCovariance(frame_length=%d, hop_length=%d, n_bands=%s, normalize=%s)"%(self.frame_length, self.hop_length, self.n_bands, self.normalize)

    @property
    def default_feature_name(self):
        return "spatial_covariance"

    @staticmethod
    def _normalize(cov: torch.Tensor) -> torch.Tensor:
        diag = torch.diagonal(cov, dim1=-2, dim2=-1).real.clamp(min=0)
        norm = torch.sqrt(diag[..., :, None] * diag[..., None, :])
        return torch.where(norm > _EPS, cov / norm.clamp(min=_EPS), torch.zeros_like(cov))

    def __call__(self, data) -> torch.Tensor:
        x = _as_tensor(data)
        frames = _frames(x, self.frame_length, self.hop_length)          # (..., C, F, L)
        frames = frames.transpose(-3, -2)                                 # (..., F, C, L)
        if self.n_bands is None:
            cov = frames @ frames.transpose(-2, -1) / self.frame_length   # (..., F, C, C)
            if self.normalize: cov = self._normalize(cov)
            return cov.movedim(-3, -1)                                    # (..., C, C, F)
        window = torch.hann_window(self.frame_length, dtype=frames.dtype, device=frames.device)
        spectrum = torch.fft.rfft(frames * window, dim=-1)               # (..., F, C, K)
        bands = np.array_split(np.arange(spectrum.shape[-1]), min(self.n_bands, spectrum.shape[-1]))
        covs = []
        for band in bands:
            s = spectrum[..., band[0]:band[-1] + 1]
            cov = s @ s.conj().transpose(-2, -1)                          # (..., F, C, C)
            covs.append(self._normalize(cov) if self.normalize else cov / self.frame_length)
        cov = torch.stack(covs, dim=-3)                                   # (..., F, B, C, C)
        cov = torch.stack([cov.real, cov.imag], dim=-4)                   # (..., F, 2, B, C, C)
        return cov.movedim(-5, -1)                                        # (..., 2, B, C, C, F)


@gin.configurable(module="ad.features")
class CrossSpectralMatrix(_FramedFeature):
    """Cross spectral matrices (CSM) between channels, per frequency bin and per frame, as used in beamforming and
    acoustic imaging. Each frame (frame_length, hop_length) is estimated with Welch's method : the frame is split in
    hann-windowed segments of n_fft samples with 50% overlap, and C[i, j] = mean(X_i conj(X_j)) over segments, with a
    one-sided power spectral density scaling (C[i, j] equals scipy.signal.csd(x_j, x_i) on the frame). Matrices are
    hermitian, and their diagonals are the power spectral densities of channels.

    Bins are restricted to [fmin, fmax] (Hz), and averaged in n_bands linear bands if n_bands is given (K = n_bands).
    Modes :
        - "complex" (default) : real and imaginary parts, shape (..., 2, K, C, C, n_frames).
        - "magnitude", "phase" : magnitude and phase (radians), shape (..., K, C, C, n_frames).
        - "coherence" : magnitude squared coherence |C_ij|^2 / (C_ii C_jj), in [0, 1], shape (..., K, C, C, n_frames).
          It is only informative with several segments per frame (frame_length >> n_fft) : it is 1 with one segment.
    The center frequencies of the K bins (or bands) are given by the frequencies attribute.
    """
    modes = ["complex", "magnitude", "phase", "coherence"]

    def __init__(self, frame_length: int = 4096, hop_length: int | None = None, n_fft: int = 1024, fmin: float = 0.,
                 fmax: float | None = None, n_bands: int | None = None, mode: str = "complex", sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, sr=sr, **kwargs)
        assert mode in self.modes, "mode must be in %s, got %s"%(self.modes, mode)
        self.n_fft = min(int(n_fft), self.frame_length)
        self.fmin, self.fmax = fmin, fmax
        self.n_bands = n_bands
        self.mode = mode
        bin_frequencies = np.arange(self.n_fft // 2 + 1) * sr / self.n_fft
        bins = np.flatnonzero((bin_frequencies >= fmin) & (bin_frequencies <= (sr / 2 if fmax is None else fmax)))
        assert len(bins) > 0, "no frequency bin in [%s, %s] Hz for n_fft=%d"%(fmin, fmax, self.n_fft)
        self._bins = bins
        bands = np.array_split(np.arange(len(bins)), min(n_bands, len(bins))) if n_bands else None
        self._bands = bands
        self.frequencies = bin_frequencies[bins] if bands is None else np.array([bin_frequencies[bins[b]].mean() for b in bands])

    def __repr__(self):
        return "CrossSpectralMatrix(mode=%s, frame_length=%d, hop_length=%d, n_fft=%d, fmin=%s, fmax=%s, n_bands=%s)"%(
            self.mode, self.frame_length, self.hop_length, self.n_fft, self.fmin, self.fmax, self.n_bands)

    @property
    def default_feature_name(self):
        return "csm"

    def __call__(self, data) -> torch.Tensor:
        x = _as_tensor(data)
        frames = _frames(x, self.frame_length, self.hop_length)                      # (..., C, F, L)
        segments = frames.unfold(-1, self.n_fft, max(1, self.n_fft // 2))           # (..., C, F, S, N)
        window = torch.hann_window(self.n_fft, dtype=frames.dtype, device=frames.device)
        spectrum = torch.fft.rfft(segments * window, dim=-1)                         # (..., C, F, S, K)
        # one-sided power spectral density scaling
        scale = torch.full((spectrum.shape[-1],), 2., dtype=frames.dtype, device=frames.device)
        scale[0] = 1.
        if self.n_fft % 2 == 0: scale[-1] = 1.
        scale = scale / (self.sr * window.square().sum())
        bins = torch.as_tensor(self._bins, device=frames.device)
        spectrum = (spectrum * scale.sqrt())[..., bins].movedim(-4, -1)              # (..., F, S, K, C)
        csm = (spectrum[..., :, None] * spectrum[..., None, :].conj()).mean(-4)       # (..., F, K, C, C)
        if self._bands is not None:
            csm = torch.stack([csm[..., b[0]:b[-1] + 1, :, :].mean(-3) for b in self._bands], dim=-3)
        if self.mode == "complex":
            return torch.stack([csm.real, csm.imag], dim=-4).movedim(-5, -1)         # (..., 2, K, C, C, F)
        if self.mode == "magnitude":
            out = csm.abs()
        elif self.mode == "phase":
            out = csm.angle()
        else:
            power = torch.diagonal(csm, dim1=-2, dim2=-1).real.clamp(min=0)
            denominator = power[..., :, None] * power[..., None, :]
            out = torch.where(denominator > _EPS ** 2, csm.abs().square() / denominator.clamp(min=_EPS ** 2), torch.zeros_like(denominator))
            out = out.clamp(max=1.)
        return out.movedim(-4, -1)                                                    # (..., K, C, C, F)


def _real_dft_matrix(n: int) -> np.ndarray:
    """orthonormal real DFT matrix (n x n) : constant, cosine and sine rows. Its first row gives sum(x) / sqrt(n)."""
    k_max = (n - 1) // 2
    t = np.arange(n)
    rows = [np.ones(n) / np.sqrt(n)]
    rows += [np.sqrt(2 / n) * np.cos(2 * np.pi * k * t / n) for k in range(1, k_max + 1)]
    if n % 2 == 0: rows.append(np.cos(np.pi * t) / np.sqrt(n))
    rows += [np.sqrt(2 / n) * np.sin(2 * np.pi * k * t / n) for k in range(1, k_max + 1)]
    return np.stack(rows)


@gin.configurable(module="ad.features")
class SpectroSpatialCovarianceVector(_FramedFeature):
    """Spectro-Spatial Covariance Vector (SSCV), from Meng et al., "Blind Estimation of Sub-band Acoustic Parameters
    from Ambisonics Recordings using Spectro-Spatial Covariance Features" (ICASSP 2025, arXiv:2411.03172). Designed
    for first-order ambisonics (M = 4 channels, 16 values per band), it applies to any number of channels.

    1. Banded covariance : hann-windowed frames (frame_length, hop_length ; by default 96 ms with 50% overlap, as in
       the paper) are transformed with a DFT, and covariance matrices are computed in n_bands mel bands (52 triangular
       filters between fmin and fmax, with weights W_b summing to 1) : Cov(n, b) = 1 / |B_b| sum_f W_b(f) X(n, f) X(n, f)^H,
       where |B_b| is the number of bins of band b.
    2. Smoothing : Cov'(n, b) = (1 - alpha) Cov(n, b) + alpha Cov'(n - 1, b), initialized with the first frame. The
       paper learns alpha jointly with the model ; here it is fixed (0 : no smoothing).
    3. Vectorization : the M diagonal entries are transformed with an orthonormal real DFT matrix, and each pair of
       conjugate off-diagonal entries (c_ij, c_ji), i < j, gives (sqrt(2) Re(c_ij), sqrt(2) Im(c_ij)). The resulting
       real vector R (M^2 values, R[0] = trace / sqrt(M)) gives SSCV = [log(R[0]), R[1] / R[0], ..., R[M^2 - 1] / R[0]].

    Output shape : (..., M^2, n_bands, n_frames). The first value is the log energy of the band, and the others are
    invariant to the gain of the signal. Silent bands give log(1e-10) and zeros.
    """

    def __init__(self, frame_length: int | None = None, hop_length: int | None = None, n_bands: int = 52, fmin: float = 0.,
                 fmax: float | None = None, alpha: float = 0., sr: int = 44100, **kwargs):
        frame_length = frame_length or int(round(0.096 * sr))
        super().__init__(frame_length=frame_length, hop_length=hop_length or frame_length // 2, sr=sr, **kwargs)
        assert 0 <= alpha < 1, "alpha must be in [0, 1), got %s"%alpha
        self.n_bands, self.fmin, self.fmax, self.alpha = n_bands, fmin, fmax, alpha
        import torchaudio
        weights = torchaudio.functional.melscale_fbanks(self.frame_length // 2 + 1, fmin, sr / 2 if fmax is None else fmax, n_bands, sr)
        counts = (weights > 0).sum(0)
        self._weights = weights / weights.sum(0).clamp(min=_EPS) / counts.clamp(min=1)    # (K, B), W_b / |B_b|

    def __repr__(self):
        return "SpectroSpatialCovarianceVector(frame_length=%d, hop_length=%d, n_bands=%d, fmin=%s, fmax=%s, alpha=%s)"%(
            self.frame_length, self.hop_length, self.n_bands, self.fmin, self.fmax, self.alpha)

    @property
    def default_feature_name(self):
        return "sscv"

    def covariance(self, data) -> torch.Tensor:
        """smoothed banded covariance matrices, (..., n_frames, n_bands, M, M) (complex)."""
        x = _as_tensor(data)
        frames = _frames(x, self.frame_length, self.hop_length)                       # (..., M, F, L)
        window = torch.hann_window(self.frame_length, dtype=frames.dtype, device=frames.device)
        spectrum = torch.fft.rfft(frames * window, dim=-1).movedim(-3, -1)           # (..., F, K, M)
        weights = self._weights.to(device=frames.device, dtype=frames.dtype).to(spectrum.dtype)
        cov = torch.einsum("...kb,...fki,...fkj->...fbij", weights, spectrum, spectrum.conj())
        if self.alpha > 0:
            smoothed = [cov[..., 0, :, :, :]]
            for n in range(1, cov.shape[-4]):
                smoothed.append((1 - self.alpha) * cov[..., n, :, :, :] + self.alpha * smoothed[-1])
            cov = torch.stack(smoothed, dim=-4)
        return cov

    def __call__(self, data) -> torch.Tensor:
        cov = self.covariance(data)                                                    # (..., F, B, M, M)
        m = cov.shape[-1]
        rdft = torch.from_numpy(_real_dft_matrix(m)).to(device=cov.device, dtype=cov.real.dtype)
        diagonal = torch.diagonal(cov, dim1=-2, dim2=-1).real @ rdft.T                 # (..., F, B, M)
        i, j = torch.triu_indices(m, m, offset=1)
        pairs = cov[..., i, j]                                                         # (..., F, B, M (M - 1) / 2)
        pairs = torch.stack([pairs.real, pairs.imag], dim=-1).flatten(-2) * np.sqrt(2)
        r = torch.cat([diagonal, pairs], dim=-1)                                       # (..., F, B, M^2)
        energy = r[..., :1]
        silent = energy <= _EPS
        ratios = torch.where(silent, torch.zeros_like(r[..., 1:]), r[..., 1:] / energy.clamp(min=_EPS))
        sscv = torch.cat([torch.log(energy.clamp(min=_EPS)), ratios], dim=-1)
        return sscv.transpose(-1, -3)                                                  # (..., M^2, B, F)



def _pairs(x: torch.Tensor) -> torch.Tensor:
    """mono inputs are compared with themselves (neutral values)."""
    x = _as_tensor(x)
    return torch.cat([x, x], dim=-2) if x.shape[-2] == 1 else x


@gin.configurable(module="ad.features")
class InterChannelLevelDifference(_FramedFeature):
    """level difference (dB) of each channel relative to the first one, per frame : (..., C - 1, n_frames) ; per linear 
    frequency band with n_bands : (..., C - 1, n_bands, n_frames). Positive values : the channel is louder than the first one."""
    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, n_bands: int | None = None, sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, sr=sr, **kwargs)
        self.n_bands = n_bands

    @property
    def default_feature_name(self): return "ild"

    def __call__(self, data):
        frames = _frames(_pairs(data), self.frame_length, self.hop_length)                 # (..., C, F, L)
        if self.n_bands is None:
            energy = frames.square().mean(-1)                                                 # (..., C, F)
        else:
            window = torch.hann_window(self.frame_length, dtype=frames.dtype, device=frames.device)
            power = torch.fft.rfft(frames * window, dim=-1).abs().square()                   # (..., C, F, K)
            bands = np.array_split(np.arange(power.shape[-1]), min(self.n_bands, power.shape[-1]))
            energy = torch.stack([power[..., b[0]:b[-1] + 1].sum(-1) for b in bands], dim=-2)   # (..., C, B, F)
        db = 10 * torch.log10(energy + _EPS)
        return db[..., 1:, :, :] - db[..., :1, :, :] if self.n_bands is not None else db[..., 1:, :] - db[..., :1, :]


class _CrossCorrelationFeature(_FramedFeature):
    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, max_delay: float = 1e-3, sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, sr=sr, **kwargs)
        self.max_delay = max_delay

    def cross_correlation(self, data):
        """normalized cross-correlations between each channel and the first one, for lags in [-max_lag, max_lag] : 
        (..., C - 1, n_frames, 2 * max_lag + 1), and lags."""
        frames = _frames(_pairs(data), self.frame_length, self.hop_length)
        L = self.frame_length
        max_lag = max(1, min(L - 1, int(round(self.max_delay * self.sr))))
        spectra = torch.fft.rfft(frames, n=2 * L)
        xcorr = torch.fft.irfft(spectra[..., 1:, :, :] * spectra[..., :1, :, :].conj(), n=2 * L)   # lag > 0 : channel delayed
        xcorr = torch.cat([xcorr[..., -max_lag:], xcorr[..., :max_lag + 1]], dim=-1)
        energy = frames.square().sum(-1)
        norm = torch.sqrt(energy[..., 1:, :] * energy[..., :1, :])
        xcorr = torch.where(norm[..., None] > _EPS, xcorr / norm[..., None].clamp(min=_EPS), torch.zeros_like(xcorr))
        return xcorr, torch.arange(-max_lag, max_lag + 1, device=frames.device, dtype=frames.dtype)


@gin.configurable(module="ad.features")
class InterChannelTimeDifference(_CrossCorrelationFeature):
    """time difference of each channel relative to the first one (lag of the maximum of their cross-correlation within 
    +/- max_delay, refined by parabolic interpolation), per frame : (..., C - 1, n_frames), in samples or seconds (unit). 
    Positive values : the channel is delayed relative to the first one."""
    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, max_delay: float = 1e-3, unit: str = "samples", 
                 sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, max_delay=max_delay, sr=sr, **kwargs)
        assert unit in ["samples", "seconds"], "unit must be samples or seconds"
        self.unit = unit

    @property
    def default_feature_name(self): return "itd"

    def __call__(self, data):
        xcorr, lags = self.cross_correlation(data)
        position, peak = _parabolic_peak(xcorr)
        delay = torch.where(peak > _EPS, position + lags[0], torch.zeros_like(peak))
        return delay / self.sr if self.unit == "seconds" else delay


def _parabolic_peak(xcorr: torch.Tensor):
    """fractional index of the maximum of xcorr along its last axis (refined by parabolic interpolation), and its value."""
    index = xcorr.argmax(-1)
    n = xcorr.shape[-1]
    left, mid, right = [xcorr.gather(-1, (index + o).clamp(0, n - 1)[..., None])[..., 0] for o in (-1, 0, 1)]
    denominator = left - 2 * mid + right
    offset = torch.where(denominator.abs() > _EPS, 0.5 * (left - right) / denominator, torch.zeros_like(mid)).clamp(-0.5, 0.5)
    return index + offset, mid


@gin.configurable(module="ad.features")
class InterChannelCoherence(_CrossCorrelationFeature):
    """maximum absolute normalized cross-correlation of each channel with the first one within +/- max_delay, per frame : 
    (..., C - 1, n_frames), in [0, 1] (1 for identical or delayed channels, ~0 for uncorrelated channels)."""
    @property
    def default_feature_name(self): return "coherence"

    def __call__(self, data):
        xcorr, _ = self.cross_correlation(data)
        return xcorr.abs().amax(-1).clamp(max=1.)


@gin.configurable(module="ad.features")
class GeneralizedCrossCorrelation(_FramedFeature):
    """Generalized cross-correlation (GCC, Knapp & Carter, "The generalized correlation method for estimation of time
    delay", 1976) between pairs of channels, per frame, for time delay estimation (TDOA) and sound source localization.
    Each frame (frame_length, hop_length) is split in hann-windowed segments of n_fft samples (frame_length by default)
    with 50% overlap, and cross / auto spectra G_ij = mean(X_j conj(X_i)), G_ii, G_jj are averaged over segments. The
    GCC is the inverse transform of W G_ij, with weighting W :
        - "none" : 1 (cross-correlation, normalized by channel energies, in [-1, 1]).
        - "roth" : 1 / G_ii (Roth processor, normalized by the reference channel of the pair).
        - "scot" : 1 / sqrt(G_ii G_jj) (smoothed coherence transform).
        - "phat" (default) : 1 / |G_ij|^beta (phase transform ; beta < 1 gives PHAT-beta). The peak is ~1 for delayed
          channels.
        - "ht" : gamma^2 / ((1 - gamma^2) |G_ij|), with gamma^2 = |G_ij|^2 / (G_ii G_jj) (Hannan-Thomson / maximum
          likelihood ; gamma^2 is clamped to 1 - 1e-3).
    With one segment per frame (n_fft = frame_length), gamma^2 = 1 : "scot" equals "phat", and "ht" is "phat" up to a
    constant ; "roth", "scot" and "ht" are only meaningful with several segments per frame (frame_length >> n_fft).

    Spectra are zero-padded to 2 * n_fft (linear correlation), and to 2 * interp * n_fft to compute the GCC with a
    resolution of 1 / interp sample. Lags are restricted to +/- max_delay (seconds ; None : n_fft - 1 samples), and
    pairs are each channel with the first one (pairs="reference", P = C - 1) or all pairs i < j (pairs="all",
    P = C (C - 1) / 2, ordered as torch.triu_indices). Positive lags : channel j is delayed relative to channel i.
    Modes :
        - "delay" (default) : lag of the maximum of the GCC (refined by parabolic interpolation), shape (..., P, n_frames),
          in samples or seconds (unit). Silent frames give 0.
        - "correlation" : the GCC, shape (..., P, n_lags, n_frames), for lags given by the lags attribute (in unit).
    Mono inputs are compared with themselves.
    """
    weightings = ["none", "roth", "scot", "phat", "ht"]
    modes = ["delay", "correlation"]

    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, n_fft: int | None = None, weighting: str = "phat",
                 beta: float = 1., max_delay: float | None = 1e-3, interp: int = 1, pairs: str = "reference", mode: str = "delay",
                 unit: str = "samples", sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, sr=sr, **kwargs)
        assert weighting in self.weightings, "weighting must be in %s, got %s"%(self.weightings, weighting)
        assert mode in self.modes, "mode must be in %s, got %s"%(self.modes, mode)
        assert pairs in ["reference", "all"], "pairs must be reference or all, got %s"%pairs
        assert unit in ["samples", "seconds"], "unit must be samples or seconds"
        assert int(interp) >= 1, "interp must be >= 1, got %s"%interp
        self.n_fft = min(int(n_fft or self.frame_length), self.frame_length)
        self.weighting, self.beta, self.max_delay, self.interp = weighting, beta, max_delay, int(interp)
        self.pairs, self.mode, self.unit = pairs, mode, unit
        max_lag = (self.n_fft - 1) * self.interp
        if max_delay is not None:
            max_lag = max(1, min(max_lag, int(round(max_delay * sr * self.interp))))
        self._max_lag = max_lag
        lags = np.arange(-max_lag, max_lag + 1) / self.interp
        self.lags = lags / sr if unit == "seconds" else lags

    def __repr__(self):
        return "GeneralizedCrossCorrelation(weighting=%s, mode=%s, frame_length=%d, hop_length=%d, n_fft=%d, max_delay=%s, interp=%d, pairs=%s)"%(
            self.weighting, self.mode, self.frame_length, self.hop_length, self.n_fft, self.max_delay, self.interp, self.pairs)

    @property
    def default_feature_name(self):
        return "gcc"

    def _pair_indices(self, n_channels: int, device):
        if self.pairs == "reference":
            j = torch.arange(1, n_channels, device=device)
            return torch.zeros_like(j), j
        i, j = torch.triu_indices(n_channels, n_channels, offset=1, device=device)
        return i, j

    @staticmethod
    def _inverse(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > _EPS, 1 / x.clamp(min=_EPS), torch.zeros_like(x))

    def _weights(self, cross: torch.Tensor, auto_i: torch.Tensor, auto_j: torch.Tensor) -> torch.Tensor:
        if self.weighting == "roth":
            return self._inverse(auto_i)
        if self.weighting == "scot":
            return self._inverse(torch.sqrt(auto_i * auto_j))
        if self.weighting == "phat":
            return self._inverse(cross.abs().pow(self.beta))
        coherence = (cross.abs().square() * self._inverse(auto_i * auto_j)).clamp(max=1 - 1e-3)
        return coherence / (1 - coherence) * self._inverse(cross.abs())

    def correlation(self, data) -> torch.Tensor:
        """GCC of each pair of channels, (..., P, n_frames, n_lags)."""
        frames = _frames(_pairs(data), self.frame_length, self.hop_length)                 # (..., C, F, L)
        segments = frames.unfold(-1, self.n_fft, max(1, self.n_fft // 2))                 # (..., C, F, S, N)
        segments = segments * torch.hann_window(self.n_fft, dtype=frames.dtype, device=frames.device)
        spectrum = torch.fft.rfft(segments, n=2 * self.n_fft)                              # (..., C, F, S, K)
        i, j = self._pair_indices(frames.shape[-3], frames.device)
        spectrum_i, spectrum_j = spectrum.index_select(-4, i), spectrum.index_select(-4, j)
        cross = (spectrum_j * spectrum_i.conj()).mean(-2)                                  # (..., P, F, K)
        if self.weighting == "none":
            energy = segments.square().sum(-1).mean(-1)                                     # (..., C, F)
            norm = torch.sqrt(energy.index_select(-2, i) * energy.index_select(-2, j))
            weighted = cross * self._inverse(norm)[..., None]
        else:
            auto_i, auto_j = spectrum_i.abs().square().mean(-2), spectrum_j.abs().square().mean(-2)
            weighted = cross * self._weights(cross, auto_i, auto_j)
        n = 2 * self.n_fft * self.interp
        gcc = torch.fft.irfft(weighted, n=n) * self.interp                                # lag > 0 : channel j delayed
        return torch.cat([gcc[..., -self._max_lag:], gcc[..., :self._max_lag + 1]], dim=-1)

    def __call__(self, data) -> torch.Tensor:
        gcc = self.correlation(data)
        if self.mode == "correlation":
            return gcc.transpose(-1, -2)                                                    # (..., P, n_lags, F)
        position, peak = _parabolic_peak(gcc)
        delay = torch.where(peak > _EPS, (position - self._max_lag) / self.interp, torch.zeros_like(peak))
        return delay / self.sr if self.unit == "seconds" else delay


@gin.configurable(module="ad.features")
class Pan(_FramedFeature):
    """stereo position per frame, from the energies of the first two channels : (E_1 - E_0) / (E_1 + E_0), in [-1, 1] 
    (-1 : left only, 0 : center, 1 : right only ; 0 for mono and silent frames). Output (..., n_frames)."""
    @property
    def default_feature_name(self): return "pan"

    def __call__(self, data):
        frames = _frames(_pairs(data), self.frame_length, self.hop_length)
        energy = frames.square().mean(-1)
        left, right = energy[..., 0, :], energy[..., 1, :]
        total = left + right
        return torch.where(total > _EPS, (right - left) / total.clamp(min=_EPS), torch.zeros_like(total))


__all__ = ['Width', 'SpatialCovariance', 'CrossSpectralMatrix', 'SpectroSpatialCovarianceVector', 'InterChannelLevelDifference', 'InterChannelTimeDifference', 'InterChannelCoherence', 'GeneralizedCrossCorrelation', 'Pan']
