"""Pseudo quadrature mirror filter (PQMF) banks : near perfect reconstruction cosine-modulated filter banks, that split
signals into critically sampled sub-bands (as used in multi-band neural vocoders and RAVE)."""
import math
import gin
import numpy as np
import torch
from scipy import signal
from scipy.optimize import fmin

from .framed import _as_tensor, _fit_length, _WaveformFeature


def _kaiser_lowpass(cutoff: float, attenuation: float, length: int | None = None) -> np.ndarray:
    """kaiser-windowed lowpass filter of cutoff (radians), with a stopband attenuation (dB)."""
    n, beta = signal.kaiserord(attenuation, cutoff / np.pi)
    n = length if length is not None else 2 * (n // 2) + 1
    return signal.firwin(n, cutoff, window=("kaiser", beta), scale=False, fs=2 * np.pi)


def pqmf_prototype(n_bands: int, attenuation: float = 100., length: int | None = None) -> np.ndarray:
    """prototype lowpass filter of a PQMF bank of n_bands bands : a kaiser-windowed filter (stopband attenuation in dB),
    whose cutoff minimizes the aliasing between bands (Lin & Vaidyanathan, 1998)."""
    def aliasing(cutoff):
        h = _kaiser_lowpass(float(np.asarray(cutoff).reshape(-1)[0]), attenuation, length)
        g = np.convolve(h, h[::-1], "full")
        return np.abs(g[g.shape[-1] // 2::2 * n_bands][1:]).max()
    cutoff = float(fmin(aliasing, 1 / n_bands, disp=0)[0])
    return _kaiser_lowpass(cutoff, attenuation, length)


def pqmf_filters(n_bands: int, attenuation: float = 100., length: int | None = None) -> np.ndarray:
    """analysis filters (n_bands, filter_length) of a PQMF bank : cosine modulations of the prototype."""
    h = pqmf_prototype(n_bands, attenuation, length)
    k = np.arange(n_bands)[:, None]
    t = np.arange(h.shape[-1]) - (h.shape[-1] - 1) / 2
    return 2 * h * np.cos((2 * k + 1) * np.pi / (2 * n_bands) * t + (-1) ** k * np.pi / 4)


@gin.configurable(module="ad.features")
class PQMF(_WaveformFeature):
    """Pseudo-QMF filter bank : splits signals (..., C, T) into n_bands critically sampled sub-bands
    (..., C * n_bands, ceil(T / n_bands)), bands of the first channel first ; signals are zero-padded to a multiple of
    n_bands. With inverse=True, sub-bands (..., C * n_bands, T') are recombined into signals (..., C, T' * n_bands).

    Filters are cosine modulations of a kaiser-windowed prototype lowpass filter (stopband attenuation in dB, length
    given by the kaiser design if filter_length is None), whose cutoff is optimized to cancel aliasing between adjacent
    bands : analysis followed by synthesis reconstructs signals with an error of about -60 dB with the default
    attenuation (near perfect reconstruction), without delay (filters are centered). Band k covers the frequencies
    [k, k + 1] * sr / (2 * n_bands) ; odd bands are spectrally inverted by the decimation.

    invert gives signals back from sub-bands (synthesis), or sub-bands back from signals with inverse=True (analysis),
    with the near perfect reconstruction of the bank.
    """
    invertible = True

    def __init__(self, n_bands: int = 16, attenuation: float = 100., filter_length: int | None = None, inverse: bool = False,
                 sr: int = 44100, **kwargs):
        assert n_bands >= 1, "n_bands must be >= 1"
        self.n_bands, self.attenuation, self.inverse = n_bands, attenuation, inverse
        super().__init__(sr=sr, **kwargs)
        self.filters = torch.from_numpy(pqmf_filters(n_bands, attenuation, filter_length)).float()   # (M, N)
        self.filter_length = self.filters.shape[-1]

    def __repr__(self):
        return "PQMF(n_bands=%d, attenuation=%s, filter_length=%d, inverse=%s)"%(self.n_bands, self.attenuation, self.filter_length, self.inverse)

    @property
    def default_feature_name(self):
        return "pqmf_%d"%self.n_bands + ("_inverse" if self.inverse else "")

    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        """(..., C, T) -> (..., C * n_bands, ceil(T / n_bands))."""
        m = self.n_bands
        batch, (c, t) = x.shape[:-2], x.shape[-2:]
        x = torch.nn.functional.pad(x.reshape(-1, 1, t), (0, (-t) % m))
        # convolution with the analysis filters (conv1d is a correlation : filters are flipped)
        weight = self.filters.flip(-1)[:, None].to(x)
        bands = torch.nn.functional.conv1d(x, weight, padding=self.filter_length // 2)[..., ::m]
        return bands.reshape(batch + (c * m, bands.shape[-1]))

    def synthesis(self, bands: torch.Tensor) -> torch.Tensor:
        """(..., C * n_bands, T') -> (..., C, T' * n_bands)."""
        m = self.n_bands
        batch, (cm, t) = bands.shape[:-2], bands.shape[-2:]
        if cm % m != 0:
            raise ValueError("PQMF synthesis needs a multiple of n_bands (%d) channels, got %d"%(m, cm))
        upsampled = torch.zeros(bands.shape[:-1] + (t * m,), dtype=bands.dtype, device=bands.device)
        upsampled[..., ::m] = bands * m
        upsampled = upsampled.reshape(-1, m, t * m)
        # synthesis filters are the time-reversed analysis filters : correlation with the analysis filters
        weight = self.filters[None].to(bands)
        x = torch.nn.functional.conv1d(upsampled, weight, padding=self.filter_length // 2)
        return x.reshape(batch + (cm // m, t * m))

    def __call__(self, data) -> torch.Tensor:
        x = _as_tensor(data)
        return self.synthesis(x) if self.inverse else self.analysis(x)

    def invert(self, features, length: int | None = None) -> torch.Tensor:
        """signals (..., C, T) from sub-bands (..., C * n_bands, T'), cropped or zero-padded to length samples if given
        (signals are zero-padded to a multiple of n_bands by the analysis) ; sub-bands from signals with inverse=True."""
        x = _as_tensor(features)
        return self.analysis(x) if self.inverse else _fit_length(self.synthesis(x), length)


__all__ = ['PQMF', 'pqmf_prototype', 'pqmf_filters']
