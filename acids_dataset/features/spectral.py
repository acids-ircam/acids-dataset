"""Spectral shape descriptors, per frame : centroid, spread, skewness, kurtosis, rolloff, flatness, crest, flux and
contrast. Frequency descriptors are given in Hz, normalized (ratio of the Nyquist frequency) or midi (unit). Silent
frames give 0."""
import gin
import torch

from .framed import _EPS, _SpectralFeature, convert_frequency, FREQUENCY_UNITS


def _moments(power: torch.Tensor, freqs: torch.Tensor):
    """normalized spectral distribution, centroid and spread (Hz) of power spectra (..., K)."""
    total = power.sum(-1, keepdim=True)
    p = power / total.clamp(min=_EPS)
    centroid = (p * freqs).sum(-1)
    spread = torch.sqrt((p * (freqs - centroid[..., None]) ** 2).sum(-1).clamp(min=0))
    return p, centroid, spread, total[..., 0] > _EPS


class _UnitFeature(_SpectralFeature):
    units = FREQUENCY_UNITS

    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, unit: str = "hz", keep_channels: bool = False,
                 sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, keep_channels=keep_channels, sr=sr, **kwargs)
        assert unit in self.units, "unit must be in %s, got %s"%(self.units, unit)
        self.unit = unit


@gin.configurable(module="features")
class SpectralCentroid(_UnitFeature):
    """center of mass of the power spectrum (brightness)."""
    @property
    def default_feature_name(self): return "spectral_centroid"

    def compute(self, x):
        power, freqs = self.power_spectrum(x)
        _, centroid, _, active = _moments(power, freqs)
        return torch.where(active, convert_frequency(centroid, self.unit, self.sr), torch.zeros_like(centroid))


@gin.configurable(module="features")
class SpectralSpread(_UnitFeature):
    """standard deviation of the power spectrum around its centroid (bandwidth)."""
    units = ["hz", "normalized"]

    @property
    def default_feature_name(self): return "spectral_spread"

    def compute(self, x):
        power, freqs = self.power_spectrum(x)
        _, _, spread, active = _moments(power, freqs)
        return torch.where(active, convert_frequency(spread, self.unit, self.sr), torch.zeros_like(spread))


@gin.configurable(module="features")
class SpectralSkewness(_SpectralFeature):
    """asymmetry of the power spectrum around its centroid (negative : energy above the centroid)."""
    @property
    def default_feature_name(self): return "spectral_skewness"

    def compute(self, x):
        power, freqs = self.power_spectrum(x)
        p, centroid, spread, active = _moments(power, freqs)
        skewness = (p * (freqs - centroid[..., None]) ** 3).sum(-1) / spread.clamp(min=_EPS) ** 3
        return torch.where(active & (spread > _EPS), skewness, torch.zeros_like(skewness))


@gin.configurable(module="features")
class SpectralKurtosis(_SpectralFeature):
    """peakedness of the power spectrum around its centroid (3 for gaussian distributions)."""
    @property
    def default_feature_name(self): return "spectral_kurtosis"

    def compute(self, x):
        power, freqs = self.power_spectrum(x)
        p, centroid, spread, active = _moments(power, freqs)
        kurtosis = (p * (freqs - centroid[..., None]) ** 4).sum(-1) / spread.clamp(min=_EPS) ** 4
        return torch.where(active & (spread > _EPS), kurtosis, torch.zeros_like(kurtosis))


@gin.configurable(module="features")
class SpectralRolloff(_UnitFeature):
    """frequency below which a ratio (percentile, 0.85 by default) of the spectral energy lies."""
    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, percentile: float = 0.85, unit: str = "hz",
                 keep_channels: bool = False, sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, unit=unit, keep_channels=keep_channels, sr=sr, **kwargs)
        assert 0 < percentile < 1, "percentile must be in ]0, 1["
        self.percentile = percentile

    @property
    def default_feature_name(self): return "spectral_rolloff"

    def compute(self, x):
        power, freqs = self.power_spectrum(x)
        cumulative = power.cumsum(-1)
        total = cumulative[..., -1:]
        index = torch.searchsorted(cumulative.contiguous(), (self.percentile * total).contiguous()).clamp(max=len(freqs) - 1)[..., 0]
        rolloff = freqs[index]
        return torch.where(total[..., 0] > _EPS, convert_frequency(rolloff, self.unit, self.sr), torch.zeros_like(rolloff))


@gin.configurable(module="features")
class SpectralFlatness(_SpectralFeature):
    """geometric mean / arithmetic mean of the power spectrum, in [0, 1] (1 for white noise, ~0 for pure tones) ; in dB if db."""
    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, db: bool = False, keep_channels: bool = False,
                 sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, keep_channels=keep_channels, sr=sr, **kwargs)
        self.db = db

    @property
    def default_feature_name(self): return "spectral_flatness"

    def compute(self, x):
        power, _ = self.power_spectrum(x)
        power = power[..., 1:]  # without DC
        mean = power.mean(-1)
        geometric = torch.exp(torch.log(power + _EPS).mean(-1))
        flatness = torch.where(mean > _EPS, (geometric / mean.clamp(min=_EPS)).clamp(max=1.), torch.zeros_like(mean))
        return 10 * torch.log10(flatness + _EPS) if self.db else flatness


@gin.configurable(module="features")
class SpectralCrest(_SpectralFeature):
    """max / mean of the power spectrum (high for peaky spectra)."""
    @property
    def default_feature_name(self): return "spectral_crest"

    def compute(self, x):
        power, _ = self.power_spectrum(x)
        mean = power.mean(-1)
        return torch.where(mean > _EPS, power.max(-1).values / mean.clamp(min=_EPS), torch.zeros_like(mean))


@gin.configurable(module="features")
class SpectralFlux(_SpectralFeature):
    """spectral change between consecutive frames : norm of the positive differences of normalized magnitude spectra
    (0 for the first frame and for stationary signals)."""
    @property
    def default_feature_name(self): return "spectral_flux"

    def compute(self, x):
        power, _ = self.power_spectrum(x)
        magnitude = power.sqrt()
        magnitude = magnitude / magnitude.sum(-1, keepdim=True).clamp(min=_EPS)
        diff = (magnitude[..., 1:, :] - magnitude[..., :-1, :]).clamp(min=0)
        flux = diff.norm(dim=-1)
        return torch.cat([flux.new_zeros(flux.shape[:-1] + (1,)), flux], dim=-1)


@gin.configurable(module="features")
class SpectralContrast(_SpectralFeature):
    """difference (dB) between peaks and valleys of the spectrum in n_bands octave bands above fmin : mean power of the
    top and bottom quantile of each band. Output (..., n_bands, n_frames)."""
    _channel_axis = -3

    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, n_bands: int = 6, fmin: float = 200., quantile: float = 0.02,
                 keep_channels: bool = False, sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, keep_channels=keep_channels, sr=sr, **kwargs)
        self.n_bands, self.fmin, self.quantile = n_bands, fmin, quantile

    @property
    def default_feature_name(self): return "spectral_contrast"

    def compute(self, x):
        power, freqs = self.power_spectrum(x)
        edges = [0.] + [self.fmin * 2 ** i for i in range(self.n_bands)]
        edges[-1] = self.sr / 2 + 1
        contrasts = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            band = power[..., (freqs >= lo) & (freqs < hi)]
            if band.shape[-1] == 0:
                contrasts.append(torch.zeros(power.shape[:-1], device=power.device, dtype=power.dtype))
                continue
            sorted_band = band.sort(dim=-1).values
            n = max(1, int(round(self.quantile * band.shape[-1])))
            valley, peak = sorted_band[..., :n].mean(-1), sorted_band[..., -n:].mean(-1)
            contrasts.append(10 * torch.log10(peak + _EPS) - 10 * torch.log10(valley + _EPS))
        return torch.stack(contrasts, dim=-2)


__all__ = ['SpectralCentroid', 'SpectralSpread', 'SpectralSkewness', 'SpectralKurtosis', 'SpectralRolloff', 'SpectralFlatness',
           'SpectralCrest', 'SpectralFlux', 'SpectralContrast']
