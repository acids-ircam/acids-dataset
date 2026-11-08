"""Common tools of frame-based descriptors : framing, channel handling, spectra and frequency units. Descriptors are
pure PyTorch (they run on any device), compute whole batches (supports_batch), and give outputs with time last."""
import math
import numpy as np
import torch

from . import RAISE_EXC_IF_FEATURE_ERROR
from .base import AcidsDatasetFeature

_EPS = 1e-10
FREQUENCY_UNITS = ["hz", "normalized", "midi"]


def _frames(x: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    """(..., C, T) -> (..., C, n_frames, frame_length) ; inputs shorter than a frame are zero-padded to one frame."""
    if x.shape[-1] < frame_length:
        x = torch.nn.functional.pad(x, (0, frame_length - x.shape[-1]))
    return x.unfold(-1, frame_length, hop_length)


def _fit_length(x: torch.Tensor, length: int | None) -> torch.Tensor:
    """crops or zero-pads signals (..., T) to length samples (unchanged if length is None)."""
    if length is None:
        return x
    if x.shape[-1] >= length:
        return x[..., :length]
    return torch.nn.functional.pad(x, (0, length - x.shape[-1]))


def _overlap_add(frames: torch.Tensor, hop_length: int) -> torch.Tensor:
    """overlap-add of frames (..., n_frames, frame_length), every hop_length samples -> (..., (n_frames - 1) * hop_length
    + frame_length)."""
    batch, (n_frames, frame_length) = frames.shape[:-2], frames.shape[-2:]
    total = (n_frames - 1) * hop_length + frame_length
    out = torch.nn.functional.fold(frames.reshape(-1, n_frames, frame_length).transpose(-1, -2), output_size=(1, total),
                                   kernel_size=(1, frame_length), stride=(1, hop_length))
    return out.reshape(batch + (total,))


def _as_tensor(data) -> torch.Tensor:
    if not torch.is_tensor(data):
        data = torch.from_numpy(np.asarray(data))
    data = data.float()
    return data[None] if data.ndim == 1 else data


def _window(name: str | None, n: int, like: torch.Tensor) -> torch.Tensor | None:
    """analysis window : "hann", "hamming", or None (rectangular)."""
    if name is None:
        return None
    if name == "hann":
        return torch.hann_window(n, dtype=like.dtype, device=like.device)
    if name == "hamming":
        return torch.hamming_window(n, dtype=like.dtype, device=like.device)
    raise ValueError("window must be hann, hamming or None, got %s"%name)


def dct_matrix(n_out: int, n_in: int, dtype=torch.float32, device=None) -> torch.Tensor:
    """orthonormal DCT-II matrix (n_out, n_in) : M @ x gives the n_out first coefficients of scipy.fft.dct(x, norm="ortho")."""
    n = torch.arange(n_in, dtype=torch.float64)
    k = torch.arange(n_out, dtype=torch.float64)[:, None]
    m = torch.cos(math.pi / n_in * (n + 0.5) * k) * math.sqrt(2. / n_in)
    m[0] /= math.sqrt(2.)
    return m.to(dtype=dtype, device=device)


def convert_frequency(f_hz: torch.Tensor, unit: str, sr: int) -> torch.Tensor:
    """converts frequencies in Hz to "hz", "normalized" (ratio of the Nyquist frequency) or "midi" (0 for 0 Hz)."""
    if unit == "hz":
        return f_hz
    if unit == "normalized":
        return f_hz / (sr / 2)
    if unit == "midi":
        return torch.where(f_hz > 0, 69 + 12 * torch.log2(f_hz.clamp(min=_EPS) / 440.), torch.zeros_like(f_hz))
    raise ValueError("unit must be in %s, got %s"%(FREQUENCY_UNITS, unit))


class _WaveformFeature(AcidsDatasetFeature):
    """base of features computed from waveforms (..., C, T), in batches (supports_batch). Subclasses implement __call__."""
    supports_batch = True

    def __init__(self, sr: int = 44100, **kwargs):
        super().__init__(**kwargs)
        self.sr = sr

    @property
    def has_hash(self):
        return False

    def __repr__(self):
        return "%s(sr=%s)"%(type(self).__name__, self.sr)

    def from_fragment(self, fragment, write: bool = True):
        data = torch.from_numpy(np.asarray(fragment.raw_audio)).float()
        try:
            out = self(data)
        except RuntimeError as e:
            if RAISE_EXC_IF_FEATURE_ERROR: raise e
            return
        out = out.numpy()
        if write:
            fragment.put_array(self.feature_name, out)
        return out


class _FramedFeature(_WaveformFeature):
    """base of frame-based descriptors. Inputs are (..., C, T) ; outputs have time last."""

    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, sr: int = 44100, **kwargs):
        super().__init__(sr=sr, **kwargs)
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length or frame_length)

    def __repr__(self):
        return "%s(frame_length=%d, hop_length=%d, sr=%s)"%(type(self).__name__, self.frame_length, self.hop_length, self.sr)


class _ChannelFeature(_FramedFeature):
    """descriptors of the mix of channels (keep_channels=False, output (..., n_frames)), or of each channel
    (keep_channels=True, output (..., C, n_frames))."""

    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, keep_channels: bool = False, sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, sr=sr, **kwargs)
        self.keep_channels = keep_channels

    def _channels(self, data) -> torch.Tensor:
        x = _as_tensor(data)
        return x if self.keep_channels else x.mean(dim=-2, keepdim=True)

    def _output(self, out: torch.Tensor) -> torch.Tensor:
        """removes the channel axis (third last for (..., C, X, n_frames), second last otherwise) if channels are mixed."""
        return out if self.keep_channels else out.squeeze(self._channel_axis)

    _channel_axis = -2

    def __call__(self, data) -> torch.Tensor:
        return self._output(self.compute(self._channels(data)))

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class _SpectralFeature(_ChannelFeature):
    """descriptors computed on the power spectrum of hann-windowed frames (n_fft = frame_length)."""

    def power_spectrum(self, x: torch.Tensor):
        """(..., C, T) -> power spectrum (..., C, n_frames, K), and frequencies of bins (K,) in Hz."""
        frames = _frames(x, self.frame_length, self.hop_length)
        window = torch.hann_window(self.frame_length, dtype=frames.dtype, device=frames.device)
        spectrum = torch.fft.rfft(frames * window, dim=-1)
        freqs = torch.fft.rfftfreq(self.frame_length, d=1. / self.sr).to(frames.device, frames.dtype)
        return spectrum.abs().square(), freqs


__all__ = []
