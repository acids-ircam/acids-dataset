"""Onset and rhythm descriptors, without models : onset strength per frame (rectified flux of log-magnitude spectra),
onset rate and tempo per chunk."""
import math
import gin
import torch

from .framed import _EPS, _SpectralFeature


class _OnsetFeature(_SpectralFeature):
    def __init__(self, frame_length: int = 1024, hop_length: int | None = 256, keep_channels: bool = False, sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, keep_channels=keep_channels, sr=sr, **kwargs)

    def onset_strength(self, x):
        """(..., C, T) -> (..., C, n_frames) : mean of positive differences of log-magnitude spectra (0 for the first frame)."""
        power, _ = self.power_spectrum(x)
        log_magnitude = torch.log1p(1000 * power.sqrt())
        diff = (log_magnitude[..., 1:, :] - log_magnitude[..., :-1, :]).clamp(min=0).mean(-1)
        # the first frame has no previous frame (new_zeros : diff may be empty for inputs of one frame)
        return torch.cat([diff.new_zeros(diff.shape[:-1] + (1,)), diff], dim=-1)

    @property
    def frame_rate(self) -> float:
        return self.sr / self.hop_length


@gin.configurable(module="ad.features")
class OnsetStrength(_OnsetFeature):
    """onset strength per frame (rectified spectral flux of log-magnitude spectra)."""
    @property
    def default_feature_name(self): return "onset_strength"

    def compute(self, x):
        return self.onset_strength(x)


def _peaks(strength: torch.Tensor, threshold: float, min_distance: int) -> torch.Tensor:
    """boolean mask of local maxima above an adaptive threshold (mean + threshold * std of the chunk), separated by at
    least min_distance frames."""
    mean, std = strength.mean(-1, keepdim=True), strength.std(-1, keepdim=True) if strength.shape[-1] > 1 else torch.zeros_like(strength[..., :1])
    padded = torch.nn.functional.pad(strength, (min_distance, min_distance), value=-float("inf"))
    windows = padded.unfold(-1, 2 * min_distance + 1, 1)
    is_max = strength >= windows.max(-1).values
    return is_max & (strength > mean + threshold * std) & (strength > _EPS)


@gin.configurable(module="ad.features")
class OnsetRate(_OnsetFeature):
    """number of onsets per second in a chunk (peaks of onset strength above mean + threshold * std, separated by
    min_interval seconds)."""
    _channel_axis = -1

    def __init__(self, frame_length: int = 1024, hop_length: int | None = 256, threshold: float = 1., min_interval: float = 0.05,
                 keep_channels: bool = False, sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, keep_channels=keep_channels, sr=sr, **kwargs)
        self.threshold, self.min_interval = threshold, min_interval

    @property
    def default_feature_name(self): return "onset_rate"

    def compute(self, x):
        strength = self.onset_strength(x)
        peaks = _peaks(strength, self.threshold, max(1, int(round(self.min_interval * self.frame_rate))))
        duration = x.shape[-1] / self.sr
        return peaks.float().sum(-1) / max(duration, _EPS)


@gin.configurable(module="ad.features")
class Tempo(_OnsetFeature):
    """tempo of a chunk (BPM) : lag of the maximum of the autocorrelation of onset strength between min_bpm and max_bpm,
    weighted by a log-normal prior around prior_bpm (that reduces octave errors) and refined by parabolic interpolation."""
    _channel_axis = -1

    def __init__(self, frame_length: int = 1024, hop_length: int | None = 256, min_bpm: float = 40., max_bpm: float = 240.,
                 prior_bpm: float = 120., prior_width: float = 1., keep_channels: bool = False, sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, keep_channels=keep_channels, sr=sr, **kwargs)
        self.min_bpm, self.max_bpm, self.prior_bpm, self.prior_width = min_bpm, max_bpm, prior_bpm, prior_width

    @property
    def default_feature_name(self): return "tempo"

    def compute(self, x):
        strength = self.onset_strength(x)
        # smoothing makes autocorrelation peaks tolerant to periods that are not multiples of the hop length
        kernel = torch.exp(-0.5 * torch.arange(-3, 4, device=x.device, dtype=x.dtype) ** 2)
        kernel = (kernel / kernel.sum()).reshape(1, 1, -1)
        shape = strength.shape
        strength = torch.nn.functional.conv1d(strength.reshape(-1, 1, shape[-1]), kernel, padding=3).reshape(shape)
        strength = strength - strength.mean(-1, keepdim=True)
        n = strength.shape[-1]
        spectrum = torch.fft.rfft(strength, n=2 * n)
        acf = torch.fft.irfft(spectrum.abs().square(), n=2 * n)[..., :n]
        lags = torch.arange(n, device=x.device, dtype=x.dtype)
        bpm = 60 * self.frame_rate / lags.clamp(min=1)
        valid = (bpm >= self.min_bpm) & (bpm <= self.max_bpm) & (lags > 0)
        prior = torch.exp(-0.5 * (torch.log2(bpm / self.prior_bpm) / self.prior_width) ** 2)
        score = torch.where(valid, acf * prior, torch.full_like(acf, -float("inf")))
        best = score.argmax(-1)
        # parabolic interpolation of the autocorrelation around its maximum
        left, center, right = [acf.gather(-1, (best + o).clamp(0, n - 1)[..., None])[..., 0] for o in (-1, 0, 1)]
        denominator = left - 2 * center + right
        offset = torch.where(denominator.abs() > _EPS, 0.5 * (left - right) / denominator, torch.zeros_like(center)).clamp(-0.5, 0.5)
        tempo = 60 * self.frame_rate / (best.to(x.dtype) + offset).clamp(min=1)
        return torch.where(valid.any(-1) & (acf[..., 0] > _EPS), tempo, torch.zeros_like(tempo))


__all__ = ['OnsetStrength', 'OnsetRate', 'Tempo']
