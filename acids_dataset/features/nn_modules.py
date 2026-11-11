"""TorchScript-compatible torch.nn.Module versions of features (see AcidsDatasetFeature.nn_module). Modules take audio
(..., C, T) and give the outputs of the corresponding features ; modules of invertible features also have an invert
method (exported to TorchScript)."""
import math
from typing import List, Optional, Tuple

import torch



def _frames(x: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    """(..., C, T) -> (..., C, n_frames, frame_length) ; inputs shorter than a frame are zero-padded to one frame."""
    if x.shape[-1] < frame_length:
        x = torch.nn.functional.pad(x, [0, frame_length - x.shape[-1]])
    return x.unfold(-1, frame_length, hop_length)


def _fit_length(x: torch.Tensor, length: Optional[int]) -> torch.Tensor:
    if length is None:
        return x
    if x.shape[-1] >= length:
        return x[..., :length]
    return torch.nn.functional.pad(x, [0, length - x.shape[-1]])


def _overlap_add(frames: torch.Tensor, hop_length: int) -> torch.Tensor:
    """overlap-add of frames (..., n_frames, frame_length) -> (..., (n_frames - 1) * hop_length + frame_length)."""
    batch = list(frames.shape[:-2])
    n_frames, frame_length = frames.shape[-2], frames.shape[-1]
    total = (n_frames - 1) * hop_length + frame_length
    out = torch.nn.functional.fold(frames.reshape(-1, n_frames, frame_length).transpose(-1, -2), output_size=[1, total],
                                   kernel_size=[1, frame_length], stride=[1, hop_length])
    return out.reshape(batch + [total])


def _convert_frequency(f_hz: torch.Tensor, unit: str, sr: int) -> torch.Tensor:
    if unit == "hz":
        return f_hz
    if unit == "normalized":
        return f_hz / (sr / 2)
    if unit == "midi":
        return torch.where(f_hz > 0, 69 + 12 * torch.log2(f_hz.clamp(min=1e-10) / 440.), torch.zeros_like(f_hz))
    raise ValueError("unknown frequency unit")


def _gather(x: torch.Tensor, index: torch.Tensor, offset: int, size: int) -> torch.Tensor:
    return x.gather(-1, (index + offset).clamp(0, size - 1).unsqueeze(-1)).squeeze(-1)


def _window_buffer(name: Optional[str], n: int) -> torch.Tensor:
    if name is None:
        return torch.ones(n)
    if name == "hann":
        return torch.hann_window(n)
    if name == "hamming":
        return torch.hamming_window(n)
    raise ValueError("window must be hann, hamming or None, got %s"%name)


# --------------------------------------------------------------------------------------------------------------------- #
#  bases
# --------------------------------------------------------------------------------------------------------------------- #

class FeatureModule(torch.nn.Module):
    """base of feature modules computed on channels (..., C, T) : on the mix of channels (keep_channels=False, then the
    channel axis channel_axis of outputs is removed), or on each channel. Subclasses implement compute ; with
    mix_channels=False, compute receives every channel and mixes them itself."""
    keep_channels: bool
    channel_axis: int
    mix_channels: bool
    frame_length: int
    hop_length: int
    sr: int

    def __init__(self, keep_channels: bool, channel_axis: int = -2, mix_channels: bool = True, frame_length: int = 1,
                 hop_length: int = 1, sr: int = 44100):
        super().__init__()
        self.keep_channels = keep_channels
        self.channel_axis = channel_axis
        self.mix_channels = mix_channels
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.sr = int(sr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if self.mix_channels and not self.keep_channels:
            x = x.mean(dim=-2, keepdim=True)
        out = self.compute(x)
        return out if self.keep_channels else out.squeeze(self.channel_axis)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def frames(self, x: torch.Tensor) -> torch.Tensor:
        return _frames(x, self.frame_length, self.hop_length)

    def power_spectrum(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(..., C, T) -> power spectrum of hann-windowed frames (..., C, n_frames, K), and frequencies of bins (K,)."""
        frames = self.frames(x)
        window = torch.hann_window(self.frame_length, dtype=frames.dtype, device=frames.device)
        spectrum = torch.fft.rfft(frames * window, dim=-1)
        freqs = torch.fft.rfftfreq(self.frame_length, d=1. / self.sr).to(frames.device, frames.dtype)
        return spectrum.abs().square(), freqs


def _rms_db(frames: torch.Tensor) -> torch.Tensor:
    return 20 * torch.log10(torch.sqrt(frames.square().mean(-1)) + 1e-10)


# --------------------------------------------------------------------------------------------------------------------- #
#  temporal
# --------------------------------------------------------------------------------------------------------------------- #

class RMSModule(FeatureModule):
    db: bool
    min_db: float

    def __init__(self, frame_length: int, hop_length: int, db: bool, min_db: float, keep_channels: bool, sr: int):
        super().__init__(keep_channels, frame_length=frame_length, hop_length=hop_length, sr=sr)
        self.db, self.min_db = db, float(min_db)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(self.frames(x).square().mean(-1))
        return (20 * torch.log10(rms + 1e-10)).clamp(min=self.min_db) if self.db else rms


class ZeroCrossingRateModule(FeatureModule):
    unit: str

    def __init__(self, frame_length: int, hop_length: int, unit: str, keep_channels: bool, sr: int):
        super().__init__(keep_channels, frame_length=frame_length, hop_length=hop_length, sr=sr)
        self.unit = unit

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        signs = torch.sign(self.frames(x))
        crossings = (signs[..., 1:] * signs[..., :-1] < 0).float().mean(-1)
        return crossings * self.sr / 2 if self.unit == "hz" else crossings


class AttackTimeModule(FeatureModule):
    low: float
    high: float

    def __init__(self, envelope_length: int, low: float, high: float, keep_channels: bool, sr: int):
        super().__init__(keep_channels, channel_axis=-1, frame_length=envelope_length, hop_length=envelope_length, sr=sr)
        self.low, self.high = float(low), float(high)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        env = torch.sqrt(self.frames(x).square().mean(-1))
        peak_value, peak_index = torch.max(env, -1)
        positions = torch.arange(env.shape[-1], device=env.device)
        before_peak = positions <= peak_index.unsqueeze(-1)
        below_low = (env < self.low * peak_value.unsqueeze(-1)) & before_peak
        below_high = (env < self.high * peak_value.unsqueeze(-1)) & before_peak
        start = torch.where(below_low.any(-1), (below_low * positions).amax(-1), torch.zeros_like(peak_index))
        end = torch.where(below_high.any(-1), (below_high * positions).amax(-1) + 1, torch.zeros_like(peak_index))
        attack = (end - start).clamp(min=0).float() * self.hop_length / self.sr
        return torch.where(peak_value > 1e-10, attack, torch.zeros_like(attack))


class TemporalCentroidModule(FeatureModule):

    def __init__(self, envelope_length: int, keep_channels: bool, sr: int):
        super().__init__(keep_channels, channel_axis=-1, frame_length=envelope_length, hop_length=envelope_length, sr=sr)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        env = torch.sqrt(self.frames(x).square().mean(-1))
        times = (torch.arange(env.shape[-1], device=env.device, dtype=env.dtype) + 0.5) * self.hop_length / self.sr
        total = env.sum(-1)
        return torch.where(total > 1e-10, (env * times).sum(-1) / total.clamp(min=1e-10), torch.zeros_like(total))


class CrestFactorModule(FeatureModule):
    db: bool

    def __init__(self, db: bool, keep_channels: bool, sr: int):
        super().__init__(keep_channels, channel_axis=-1, sr=sr)
        self.db = db

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        peak, rms = x.abs().amax(-1), torch.sqrt(x.square().mean(-1))
        crest = torch.where(rms > 1e-10, peak / rms.clamp(min=1e-10), torch.zeros_like(rms))
        return 20 * torch.log10(crest + 1e-10) if self.db else crest


# --------------------------------------------------------------------------------------------------------------------- #
#  spectral
# --------------------------------------------------------------------------------------------------------------------- #

def _moments(power: torch.Tensor, freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total = power.sum(-1, keepdim=True)
    p = power / total.clamp(min=1e-10)
    centroid = (p * freqs).sum(-1)
    spread = torch.sqrt((p * (freqs - centroid.unsqueeze(-1)) ** 2).sum(-1).clamp(min=0))
    return p, centroid, spread, total.squeeze(-1) > 1e-10


class SpectralMomentModule(FeatureModule):
    """spectral centroid, spread, skewness and kurtosis (moment)."""
    moment: str
    unit: str

    def __init__(self, moment: str, frame_length: int, hop_length: int, unit: str, keep_channels: bool, sr: int):
        super().__init__(keep_channels, frame_length=frame_length, hop_length=hop_length, sr=sr)
        assert moment in ["centroid", "spread", "skewness", "kurtosis"]
        self.moment, self.unit = moment, unit

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        power, freqs = self.power_spectrum(x)
        p, centroid, spread, active = _moments(power, freqs)
        if self.moment == "centroid":
            return torch.where(active, _convert_frequency(centroid, self.unit, self.sr), torch.zeros_like(centroid))
        if self.moment == "spread":
            return torch.where(active, _convert_frequency(spread, self.unit, self.sr), torch.zeros_like(spread))
        exponent = 3 if self.moment == "skewness" else 4
        value = (p * (freqs - centroid.unsqueeze(-1)) ** exponent).sum(-1) / spread.clamp(min=1e-10) ** exponent
        return torch.where(active & (spread > 1e-10), value, torch.zeros_like(value))


class SpectralRolloffModule(FeatureModule):
    percentile: float
    unit: str

    def __init__(self, frame_length: int, hop_length: int, percentile: float, unit: str, keep_channels: bool, sr: int):
        super().__init__(keep_channels, frame_length=frame_length, hop_length=hop_length, sr=sr)
        self.percentile, self.unit = float(percentile), unit

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        power, freqs = self.power_spectrum(x)
        cumulative = power.cumsum(-1)
        total = cumulative[..., -1:]
        index = torch.searchsorted(cumulative.contiguous(), (self.percentile * total).contiguous()).clamp(max=freqs.shape[0] - 1)[..., 0]
        rolloff = freqs[index]
        return torch.where(total[..., 0] > 1e-10, _convert_frequency(rolloff, self.unit, self.sr), torch.zeros_like(rolloff))


class SpectralFlatnessModule(FeatureModule):
    db: bool

    def __init__(self, frame_length: int, hop_length: int, db: bool, keep_channels: bool, sr: int):
        super().__init__(keep_channels, frame_length=frame_length, hop_length=hop_length, sr=sr)
        self.db = db

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        power, _ = self.power_spectrum(x)
        power = power[..., 1:]
        mean = power.mean(-1)
        geometric = torch.exp(torch.log(power + 1e-10).mean(-1))
        flatness = torch.where(mean > 1e-10, (geometric / mean.clamp(min=1e-10)).clamp(max=1.), torch.zeros_like(mean))
        return 10 * torch.log10(flatness + 1e-10) if self.db else flatness


class SpectralCrestModule(FeatureModule):

    def __init__(self, frame_length: int, hop_length: int, keep_channels: bool, sr: int):
        super().__init__(keep_channels, frame_length=frame_length, hop_length=hop_length, sr=sr)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        power, _ = self.power_spectrum(x)
        mean = power.mean(-1)
        return torch.where(mean > 1e-10, power.amax(-1) / mean.clamp(min=1e-10), torch.zeros_like(mean))


class SpectralFluxModule(FeatureModule):

    def __init__(self, frame_length: int, hop_length: int, keep_channels: bool, sr: int):
        super().__init__(keep_channels, frame_length=frame_length, hop_length=hop_length, sr=sr)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        power, _ = self.power_spectrum(x)
        magnitude = power.sqrt()
        magnitude = magnitude / magnitude.sum(-1, keepdim=True).clamp(min=1e-10)
        diff = (magnitude[..., 1:, :] - magnitude[..., :-1, :]).clamp(min=0)
        flux = torch.linalg.vector_norm(diff, 2., [-1])
        return torch.cat([flux.new_zeros(list(flux.shape[:-1]) + [1]), flux], dim=-1)


class SpectralContrastModule(FeatureModule):
    """bands are contiguous ranges of bins [starts[i], ends[i]), whose n_valleys[i] lowest / highest bins are averaged."""
    starts: List[int]
    ends: List[int]
    n_values: List[int]

    def __init__(self, frame_length: int, hop_length: int, n_bands: int, fmin: float, quantile: float, keep_channels: bool, sr: int):
        super().__init__(keep_channels, channel_axis=-3, frame_length=frame_length, hop_length=hop_length, sr=sr)
        freqs = torch.fft.rfftfreq(frame_length, d=1. / sr)
        edges = [0.] + [fmin * 2 ** i for i in range(n_bands)]
        edges[-1] = sr / 2 + 1
        self.starts, self.ends, self.n_values = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            index = torch.nonzero((freqs >= lo) & (freqs < hi))[:, 0]
            start, end = (int(index[0]), int(index[-1]) + 1) if len(index) else (0, 0)
            self.starts.append(start)
            self.ends.append(end)
            self.n_values.append(max(1, int(round(quantile * (end - start)))))

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        power, _ = self.power_spectrum(x)
        contrasts: List[torch.Tensor] = []
        for i in range(len(self.starts)):
            if self.ends[i] == self.starts[i]:
                contrasts.append(torch.zeros(power.shape[:-1], device=power.device, dtype=power.dtype))
                continue
            sorted_band = power[..., self.starts[i]:self.ends[i]].sort(dim=-1)[0]
            n = self.n_values[i]
            valley, peak = sorted_band[..., :n].mean(-1), sorted_band[..., -n:].mean(-1)
            contrasts.append(10 * torch.log10(peak + 1e-10) - 10 * torch.log10(valley + 1e-10))
        return torch.stack(contrasts, dim=-2)


# --------------------------------------------------------------------------------------------------------------------- #
#  quality
# --------------------------------------------------------------------------------------------------------------------- #

class ClippingRatioModule(FeatureModule):
    threshold: float

    def __init__(self, threshold: float, keep_channels: bool, sr: int):
        super().__init__(keep_channels, channel_axis=-1, mix_channels=False, sr=sr)
        self.threshold = float(threshold)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        ratio = (x.abs() >= self.threshold).float().mean(-1)
        return ratio if self.keep_channels else ratio.mean(-1, keepdim=True)


class PeakModule(FeatureModule):
    db: bool

    def __init__(self, db: bool, keep_channels: bool, sr: int):
        super().__init__(keep_channels, channel_axis=-1, mix_channels=False, sr=sr)
        self.db = db

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        peak = x.abs().amax(-1)
        if not self.keep_channels:
            peak = peak.amax(-1, keepdim=True)
        return 20 * torch.log10(peak + 1e-10) if self.db else peak


class DCOffsetModule(FeatureModule):

    def __init__(self, keep_channels: bool, sr: int):
        super().__init__(keep_channels, channel_axis=-1, sr=sr)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(-1)


class DynamicRangeModule(FeatureModule):
    low: float
    high: float
    silence_db: float

    def __init__(self, frame_length: int, hop_length: int, low: float, high: float, silence_db: float, keep_channels: bool, sr: int):
        super().__init__(keep_channels, channel_axis=-1, frame_length=frame_length, hop_length=hop_length, sr=sr)
        self.low, self.high, self.silence_db = float(low), float(high), float(silence_db)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        rms_db = _rms_db(self.frames(x))
        active = rms_db > self.silence_db
        masked = torch.where(active, rms_db, torch.full_like(rms_db, float("nan")))
        high, low = torch.nanquantile(masked, self.high, dim=-1), torch.nanquantile(masked, self.low, dim=-1)
        return torch.where(active.any(-1), high - low, torch.zeros_like(high))


class SilenceRatioModule(FeatureModule):
    threshold_db: float

    def __init__(self, frame_length: int, hop_length: int, threshold_db: float, keep_channels: bool, sr: int):
        super().__init__(keep_channels, channel_axis=-1, frame_length=frame_length, hop_length=hop_length, sr=sr)
        self.threshold_db = float(threshold_db)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        return (_rms_db(self.frames(x)) < self.threshold_db).float().mean(-1)


# --------------------------------------------------------------------------------------------------------------------- #
#  onsets
# --------------------------------------------------------------------------------------------------------------------- #

class _OnsetModule(FeatureModule):

    def onset_strength(self, x: torch.Tensor) -> torch.Tensor:
        power, _ = self.power_spectrum(x)
        log_magnitude = torch.log1p(1000 * power.sqrt())
        diff = (log_magnitude[..., 1:, :] - log_magnitude[..., :-1, :]).clamp(min=0).mean(-1)
        return torch.cat([diff.new_zeros(list(diff.shape[:-1]) + [1]), diff], dim=-1)

    def frame_rate(self) -> float:
        return self.sr / self.hop_length


class OnsetStrengthModule(_OnsetModule):

    def __init__(self, frame_length: int, hop_length: int, keep_channels: bool, sr: int):
        super().__init__(keep_channels, frame_length=frame_length, hop_length=hop_length, sr=sr)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        return self.onset_strength(x)


class OnsetRateModule(_OnsetModule):
    threshold: float
    min_distance: int

    def __init__(self, frame_length: int, hop_length: int, threshold: float, min_interval: float, keep_channels: bool, sr: int):
        super().__init__(keep_channels, channel_axis=-1, frame_length=frame_length, hop_length=hop_length, sr=sr)
        self.threshold = float(threshold)
        self.min_distance = max(1, int(round(min_interval * sr / hop_length)))

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        strength = self.onset_strength(x)
        md = self.min_distance
        mean = strength.mean(-1, keepdim=True)
        std = strength.std(-1, keepdim=True) if strength.shape[-1] > 1 else torch.zeros_like(strength[..., :1])
        padded = torch.nn.functional.pad(strength, [md, md], value=-float("inf"))
        is_max = strength >= padded.unfold(-1, 2 * md + 1, 1).amax(-1)
        peaks = is_max & (strength > mean + self.threshold * std) & (strength > 1e-10)
        duration = x.shape[-1] / self.sr
        return peaks.float().sum(-1) / max(duration, 1e-10)


class TempoModule(_OnsetModule):
    min_bpm: float
    max_bpm: float
    prior_bpm: float
    prior_width: float

    def __init__(self, frame_length: int, hop_length: int, min_bpm: float, max_bpm: float, prior_bpm: float, prior_width: float,
                 keep_channels: bool, sr: int):
        super().__init__(keep_channels, channel_axis=-1, frame_length=frame_length, hop_length=hop_length, sr=sr)
        self.min_bpm, self.max_bpm = float(min_bpm), float(max_bpm)
        self.prior_bpm, self.prior_width = float(prior_bpm), float(prior_width)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        strength = self.onset_strength(x)
        frame_rate = self.frame_rate()
        kernel = torch.exp(-0.5 * torch.arange(-3, 4, device=x.device, dtype=x.dtype) ** 2)
        kernel = (kernel / kernel.sum()).reshape(1, 1, -1)
        shape = strength.shape
        strength = torch.nn.functional.conv1d(strength.reshape(-1, 1, shape[-1]), kernel, padding=3).reshape(shape)
        strength = strength - strength.mean(-1, keepdim=True)
        n = strength.shape[-1]
        spectrum = torch.fft.rfft(strength, n=2 * n)
        acf = torch.fft.irfft(spectrum.abs().square(), n=2 * n)[..., :n]
        lags = torch.arange(n, device=x.device, dtype=x.dtype)
        bpm = 60 * frame_rate / lags.clamp(min=1)
        valid = (bpm >= self.min_bpm) & (bpm <= self.max_bpm) & (lags > 0)
        prior = torch.exp(-0.5 * (torch.log2(bpm / self.prior_bpm) / self.prior_width) ** 2)
        score = torch.where(valid, acf * prior, torch.full_like(acf, -float("inf")))
        best = score.argmax(-1)
        left, center, right = _gather(acf, best, -1, n), _gather(acf, best, 0, n), _gather(acf, best, 1, n)
        denominator = left - 2 * center + right
        offset = torch.where(denominator.abs() > 1e-10, 0.5 * (left - right) / denominator, torch.zeros_like(center)).clamp(-0.5, 0.5)
        tempo = 60 * frame_rate / (best.to(x.dtype) + offset).clamp(min=1)
        return torch.where(valid.any(-1) & (acf[..., 0] > 1e-10), tempo, torch.zeros_like(tempo))


# --------------------------------------------------------------------------------------------------------------------- #
#  harmonic
# --------------------------------------------------------------------------------------------------------------------- #



class ChromaModule(FeatureModule):
    fmin: float
    fmax: float
    norm: Optional[str]

    def __init__(self, frame_length: int, hop_length: int, fmin: float, fmax: float, norm: Optional[str], keep_channels: bool, sr: int):
        super().__init__(keep_channels, channel_axis=-3, frame_length=frame_length, hop_length=hop_length, sr=sr)
        self.fmin, self.fmax, self.norm = float(fmin), float(min(fmax, sr / 2)), norm

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        power, freqs = self.power_spectrum(x)
        valid = (freqs >= self.fmin) & (freqs <= self.fmax)
        classes = torch.remainder(torch.round(12 * torch.log2(freqs.clamp(min=1e-10) / 16.351597831287414)), 12).long()
        mapping = torch.nn.functional.one_hot(classes, 12).to(power.dtype) * valid.unsqueeze(-1).to(power.dtype)
        chroma = (power @ mapping).transpose(-1, -2)
        norm = self.norm
        if norm is not None:
            if norm == "max":
                chroma = chroma / chroma.amax(-2, keepdim=True).clamp(min=1e-10)
            elif norm == "sum":
                chroma = chroma / chroma.sum(-2, keepdim=True).clamp(min=1e-10)
        return chroma


def _autocorrelation(frames: torch.Tensor) -> torch.Tensor:
    L = frames.shape[-1]
    spectrum = torch.fft.rfft(frames, n=2 * L)
    acf = torch.fft.irfft(spectrum.abs().square(), n=2 * L)[..., :L]
    lags = torch.arange(L, device=frames.device, dtype=frames.dtype)
    acf = acf * L / (L - lags).clamp(min=1)
    return acf / acf[..., :1].clamp(min=1e-10)


def _periodicity(frames: torch.Tensor, sr: int, fmin: float, fmax: float) -> Tuple[torch.Tensor, torch.Tensor]:
    acf = _autocorrelation(frames)
    L = frames.shape[-1]
    lo, hi = max(1, int(math.floor(sr / fmax))), min(L - 2, int(math.ceil(sr / fmin)))
    window = acf[..., lo:hi + 1]
    value = window.amax(-1)
    index = (window >= 0.9 * value.unsqueeze(-1)).float().argmax(-1)
    size = window.shape[-1]
    for _ in range(size):
        current = _gather(window, index, 0, size)
        following = _gather(window, index.clamp(max=size - 1), 1, size)
        grow = following > current
        if not bool(grow.any()):
            break
        index = index + grow.long()
    best = index + lo
    left, center, right = _gather(acf, best, -1, L), _gather(acf, best, 0, L), _gather(acf, best, 1, L)
    denominator = left - 2 * center + right
    offset = torch.where(denominator.abs() > 1e-10, 0.5 * (left - right) / denominator, torch.zeros_like(value)).clamp(-0.5, 0.5)
    energy = frames.square().sum(-1)
    value = torch.where(energy > 1e-10, value.clamp(0, 1), torch.zeros_like(value))
    return value, best.to(frames.dtype) + offset


class HarmonicityModule(FeatureModule):
    fmin: float
    fmax: float

    def __init__(self, frame_length: int, hop_length: int, fmin: float, fmax: float, keep_channels: bool, sr: int):
        super().__init__(keep_channels, frame_length=frame_length, hop_length=hop_length, sr=sr)
        self.fmin, self.fmax = float(fmin), float(fmax)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        return _periodicity(self.frames(x), self.sr, self.fmin, self.fmax)[0]


class InharmonicityModule(FeatureModule):
    n_harmonics: int
    fmin: float
    fmax: float

    def __init__(self, frame_length: int, hop_length: int, n_harmonics: int, fmin: float, fmax: float, keep_channels: bool, sr: int):
        super().__init__(keep_channels, frame_length=frame_length, hop_length=hop_length, sr=sr)
        self.n_harmonics, self.fmin, self.fmax = int(n_harmonics), float(fmin), float(fmax)

    def _peak(self, power: torch.Tensor, log_power: torch.Tensor, freqs: torch.Tensor, center: torch.Tensor,
              radius: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = (freqs - center.unsqueeze(-1)).abs() <= radius.unsqueeze(-1)
        masked = torch.where(mask, power, torch.full_like(power, -1.))
        index = masked.argmax(-1)
        K = power.shape[-1]
        left, mid, right = _gather(log_power, index, -1, K), _gather(log_power, index, 0, K), _gather(log_power, index, 1, K)
        denominator = left - 2 * mid + right
        offset = torch.where(denominator.abs() > 1e-10, 0.5 * (left - right) / denominator, torch.zeros_like(mid)).clamp(-0.5, 0.5)
        df = freqs[1] - freqs[0]
        frequency = freqs[index] + offset * df
        return frequency, _gather(power, index, 0, K), mask.any(-1)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        power, freqs = self.power_spectrum(x)
        log_power = torch.log(power + 1e-10)
        periodicity, lag = _periodicity(self.frames(x), self.sr, self.fmin, self.fmax)
        f0 = self.sr / lag.clamp(min=1)
        f1, _, _ = self._peak(power, log_power, freqs, f0, 0.25 * f0)
        deviations: List[torch.Tensor] = []
        weights: List[torch.Tensor] = []
        for k in range(2, self.n_harmonics + 1):
            center = k * f1
            fk, pk, found = self._peak(power, log_power, freqs, center, 0.25 * f1)
            inside = found & (center < self.sr / 2)
            deviations.append(torch.where(inside, (fk / center.clamp(min=1e-10) - 1).abs(), torch.zeros_like(fk)))
            weights.append(torch.where(inside, pk, torch.zeros_like(pk)))
        deviation, weight = torch.stack(deviations, -1), torch.stack(weights, -1)
        total = weight.sum(-1)
        inharmonicity = (deviation * weight).sum(-1) / total.clamp(min=1e-10)
        return torch.where((total > 1e-10) & (periodicity > 0), inharmonicity, torch.zeros_like(inharmonicity))


# --------------------------------------------------------------------------------------------------------------------- #
#  representations
# --------------------------------------------------------------------------------------------------------------------- #

class _WindowedModule(FeatureModule):
    has_window: bool

    def __init__(self, frame_length: int, hop_length: int, window: Optional[str], keep_channels: bool, sr: int):
        super().__init__(keep_channels, channel_axis=-3, frame_length=frame_length, hop_length=hop_length, sr=sr)
        self.has_window = window is not None
        self.register_buffer("window", _window_buffer(window, frame_length))

    def windowed_frames(self, x: torch.Tensor) -> torch.Tensor:
        frames = self.frames(x)
        return frames * self.window.to(frames) if self.has_window else frames


class DCTModule(_WindowedModule):
    """DCT of windowed frames (forward), and its inverse by weighted overlap-add (invert)."""

    def __init__(self, frame_length: int, hop_length: int, matrix: torch.Tensor, window: Optional[str], keep_channels: bool, sr: int):
        super().__init__(frame_length, hop_length, window, keep_channels, sr)
        self.register_buffer("matrix", matrix.float())

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        frames = self.windowed_frames(x)
        return (frames @ self.matrix.to(frames).T).transpose(-2, -1)

    @torch.jit.export
    def invert(self, y: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        if not self.keep_channels:
            y = y.unsqueeze(-3)
        frames = y.transpose(-2, -1) @ self.matrix.to(y)
        window = self.window.to(frames)
        x = _overlap_add(frames * window, self.hop_length)
        norm = _overlap_add(window.square().expand(frames.shape[-2], -1), self.hop_length)
        x = torch.where(norm > 1e-6, x / norm.clamp(min=1e-6), torch.zeros_like(x))
        if not self.keep_channels:
            x = x.squeeze(-2)
        return _fit_length(x, length)


class GammatoneModule(_WindowedModule):
    db: bool

    def __init__(self, frame_length: int, hop_length: int, weights: torch.Tensor, db: bool, window: Optional[str], keep_channels: bool, sr: int):
        super().__init__(frame_length, hop_length, window, keep_channels, sr)
        self.register_buffer("weights", weights.float())
        self.db = db

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        power = torch.fft.rfft(self.windowed_frames(x), dim=-1).abs().square()
        bands = (power @ self.weights.to(power).T).transpose(-2, -1)
        return 10 * torch.log10(bands + 1e-10) if self.db else bands


class CepstrumModule(_WindowedModule):
    n_coeffs: int

    def __init__(self, frame_length: int, hop_length: int, n_coeffs: int, window: Optional[str], keep_channels: bool, sr: int):
        super().__init__(frame_length, hop_length, window, keep_channels, sr)
        self.n_coeffs = int(n_coeffs)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        log_magnitude = torch.log(torch.fft.rfft(self.windowed_frames(x), dim=-1).abs() + 1e-10)
        cepstrum = torch.fft.irfft(log_magnitude, n=self.frame_length, dim=-1)
        return cepstrum[..., :self.n_coeffs].transpose(-2, -1)


class PQMFModule(torch.nn.Module):
    """PQMF analysis (forward) and synthesis (invert) ; swapped with inverse=True (see features.PQMF)."""
    n_bands: int
    filter_length: int
    inverse: bool

    def __init__(self, filters: torch.Tensor, inverse: bool):
        super().__init__()
        self.register_buffer("filters", filters.float())
        self.n_bands = int(filters.shape[0])
        self.filter_length = int(filters.shape[-1])
        self.inverse = inverse

    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        m = self.n_bands
        batch = list(x.shape[:-2])
        c, t = x.shape[-2], x.shape[-1]
        x = torch.nn.functional.pad(x.reshape(-1, 1, t), [0, (-t) % m])
        weight = self.filters.flip(-1).unsqueeze(1).to(x)
        bands = torch.nn.functional.conv1d(x, weight, padding=self.filter_length // 2)[..., ::m]
        return bands.reshape(batch + [c * m, bands.shape[-1]])

    def synthesis(self, bands: torch.Tensor) -> torch.Tensor:
        m = self.n_bands
        batch = list(bands.shape[:-2])
        cm, t = bands.shape[-2], bands.shape[-1]
        if cm % m != 0:
            raise ValueError("PQMF synthesis needs a multiple of n_bands channels")
        upsampled = torch.zeros(list(bands.shape[:-1]) + [t * m], dtype=bands.dtype, device=bands.device)
        upsampled[..., ::m] = bands * m
        upsampled = upsampled.reshape(-1, m, t * m)
        weight = self.filters.unsqueeze(0).to(bands)
        x = torch.nn.functional.conv1d(upsampled, weight, padding=self.filter_length // 2)
        return x.reshape(batch + [cm // m, t * m])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.synthesis(x) if self.inverse else self.analysis(x)

    @torch.jit.export
    def invert(self, x: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.analysis(x) if self.inverse else _fit_length(self.synthesis(x), length)


__all__ = ['FeatureModule', 'RMSModule', 'ZeroCrossingRateModule', 'AttackTimeModule', 'TemporalCentroidModule',
           'CrestFactorModule', 'SpectralMomentModule', 'SpectralRolloffModule', 'SpectralFlatnessModule',
           'SpectralCrestModule', 'SpectralFluxModule', 'SpectralContrastModule', 'ClippingRatioModule', 'PeakModule',
           'DCOffsetModule', 'DynamicRangeModule', 'SilenceRatioModule', 'OnsetStrengthModule', 'OnsetRateModule',
           'TempoModule', 'ChromaModule', 'HarmonicityModule', 'InharmonicityModule', 'DCTModule', 'GammatoneModule',
           'CepstrumModule', 'PQMFModule']
