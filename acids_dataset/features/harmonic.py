"""Harmonic descriptors per frame : chroma, harmonicity (periodicity) and inharmonicity."""
import math
import gin
import torch

from .framed import _EPS, _frames, _ChannelFeature, _SpectralFeature

C0 = 16.351597831287414


@gin.configurable(module="features")
class Chroma(_SpectralFeature):
    """energy of the 12 pitch classes (0 : C, 1 : C#, ... 11 : B) between fmin and fmax, per frame, normalized by the
    maximum of each frame (norm="max"), by their sum (norm="sum"), or not (norm=None). Output (..., 12, n_frames)."""
    _channel_axis = -3

    def __init__(self, frame_length: int = 4096, hop_length: int | None = 1024, fmin: float = 32.7, fmax: float = 4186.,
                 norm: str | None = "max", keep_channels: bool = False, sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, keep_channels=keep_channels, sr=sr, **kwargs)
        assert norm in ["max", "sum", None], "norm must be max, sum or None"
        self.fmin, self.fmax, self.norm = fmin, fmax, norm

    @property
    def default_feature_name(self): return "chroma"

    def compute(self, x):
        power, freqs = self.power_spectrum(x)
        valid = (freqs >= self.fmin) & (freqs <= min(self.fmax, self.sr / 2))
        classes = torch.remainder(torch.round(12 * torch.log2(freqs.clamp(min=_EPS) / C0)), 12).long()
        mapping = torch.nn.functional.one_hot(classes, 12).to(power.dtype) * valid[:, None].to(power.dtype)   # (K, 12)
        chroma = (power @ mapping).transpose(-1, -2)                                                         # (..., C, 12, n_frames)
        if self.norm == "max":
            chroma = chroma / chroma.amax(-2, keepdim=True).clamp(min=_EPS)
        elif self.norm == "sum":
            chroma = chroma / chroma.sum(-2, keepdim=True).clamp(min=_EPS)
        return chroma


def _autocorrelation(frames: torch.Tensor) -> torch.Tensor:
    """unbiased normalized autocorrelation of frames (..., L), over lags [0, L)."""
    L = frames.shape[-1]
    spectrum = torch.fft.rfft(frames, n=2 * L)
    acf = torch.fft.irfft(spectrum.abs().square(), n=2 * L)[..., :L]
    lags = torch.arange(L, device=frames.device, dtype=frames.dtype)
    acf = acf * L / (L - lags).clamp(min=1)
    return acf / acf[..., :1].clamp(min=_EPS)


def _periodicity(frames: torch.Tensor, sr: int, fmin: float, fmax: float):
    """maximum of the normalized autocorrelation between lags sr / fmax and sr / fmin, and its (interpolated) lag."""
    acf = _autocorrelation(frames)
    L = frames.shape[-1]
    lo, hi = max(1, int(math.floor(sr / fmax))), min(L - 2, int(math.ceil(sr / fmin)))
    window = acf[..., lo:hi + 1]
    value = window.max(-1).values
    # octave errors : the first lag reaching 90% of the maximum is taken (multiples of the period are as correlated), 
    # then moved to its local maximum
    index = (window >= 0.9 * value[..., None]).float().argmax(-1)
    for _ in range(window.shape[-1]):
        current = window.gather(-1, index[..., None])[..., 0]
        following = window.gather(-1, (index + 1).clamp(max=window.shape[-1] - 1)[..., None])[..., 0]
        grow = following > current
        if not bool(grow.any()):
            break
        index = index + grow.long()
    best = index + lo
    left, center, right = [acf.gather(-1, (best + o).clamp(0, L - 1)[..., None])[..., 0] for o in (-1, 0, 1)]
    denominator = left - 2 * center + right
    offset = torch.where(denominator.abs() > _EPS, 0.5 * (left - right) / denominator, torch.zeros_like(value)).clamp(-0.5, 0.5)
    energy = frames.square().sum(-1)
    value = torch.where(energy > _EPS, value.clamp(0, 1), torch.zeros_like(value))
    return value, best.to(frames.dtype) + offset


@gin.configurable(module="features")
class Harmonicity(_ChannelFeature):
    """periodicity of frames : maximum of the normalized autocorrelation for periods between 1 / fmax and 1 / fmin, in
    [0, 1] (~1 for periodic sounds, ~0 for noise)."""
    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, fmin: float = 50., fmax: float = 1000.,
                 keep_channels: bool = False, sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, keep_channels=keep_channels, sr=sr, **kwargs)
        self.fmin, self.fmax = fmin, fmax

    @property
    def default_feature_name(self): return "harmonicity"

    def compute(self, x):
        return _periodicity(_frames(x, self.frame_length, self.hop_length), self.sr, self.fmin, self.fmax)[0]


@gin.configurable(module="features")
class Inharmonicity(_SpectralFeature):
    """deviation of partials from a harmonic series : the fundamental is estimated by autocorrelation and refined on the
    first spectral peak f_1, and inharmonicity is the mean relative deviation |f_k / (k f_1) - 1| of the peaks of partials
    k = 2 ... n_harmonics, weighted by their power (~0 for harmonic sounds). Frames without fundamental give 0."""
    def __init__(self, frame_length: int = 4096, hop_length: int | None = None, n_harmonics: int = 10, fmin: float = 50., fmax: float = 1000.,
                 keep_channels: bool = False, sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, keep_channels=keep_channels, sr=sr, **kwargs)
        self.n_harmonics, self.fmin, self.fmax = n_harmonics, fmin, fmax

    @property
    def default_feature_name(self): return "inharmonicity"

    def _peak(self, power, log_power, freqs, center, radius):
        """interpolated frequency and power of the highest bin within center +/- radius (Hz)."""
        mask = (freqs - center[..., None]).abs() <= radius[..., None]
        masked = torch.where(mask, power, torch.full_like(power, -1.))
        index = masked.argmax(-1)
        K = power.shape[-1]
        left, mid, right = [log_power.gather(-1, (index + o).clamp(0, K - 1)[..., None])[..., 0] for o in (-1, 0, 1)]
        denominator = left - 2 * mid + right
        offset = torch.where(denominator.abs() > _EPS, 0.5 * (left - right) / denominator, torch.zeros_like(mid)).clamp(-0.5, 0.5)
        df = freqs[1] - freqs[0]
        frequency = freqs[index] + offset * df
        return frequency, power.gather(-1, index[..., None])[..., 0], mask.any(-1)

    def compute(self, x):
        power, freqs = self.power_spectrum(x)
        log_power = torch.log(power + _EPS)
        periodicity, lag = _periodicity(_frames(x, self.frame_length, self.hop_length), self.sr, self.fmin, self.fmax)
        f0 = self.sr / lag.clamp(min=1)
        f1, _, _ = self._peak(power, log_power, freqs, f0, 0.25 * f0)
        deviations, weights = [], []
        for k in range(2, self.n_harmonics + 1):
            center = k * f1
            fk, pk, found = self._peak(power, log_power, freqs, center, 0.25 * f1)
            inside = found & (center < self.sr / 2)
            deviations.append(torch.where(inside, (fk / center.clamp(min=_EPS) - 1).abs(), torch.zeros_like(fk)))
            weights.append(torch.where(inside, pk, torch.zeros_like(pk)))
        deviations, weights = torch.stack(deviations, -1), torch.stack(weights, -1)
        total = weights.sum(-1)
        inharmonicity = (deviations * weights).sum(-1) / total.clamp(min=_EPS)
        return torch.where((total > _EPS) & (periodicity > 0), inharmonicity, torch.zeros_like(inharmonicity))


__all__ = ['Chroma', 'Harmonicity', 'Inharmonicity']
