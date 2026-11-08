"""Spectral representations : short-time and full Fourier transforms, discrete cosine transform, gammatone
spectrogram and real cepstrum. Framed outputs are (..., n_coefficients, n_frames) for the mix of channels, or
(..., C, n_coefficients, n_frames) with keep_channels."""
import math
import gin
import torch
import acids_transforms as at
from absl import logging
from typing import Optional

from .framed import _EPS, _frames, _window, _as_tensor, _fit_length, _overlap_add, _ChannelFeature, _WaveformFeature, dct_matrix


class _RepresentationFeature(_ChannelFeature):
    _channel_axis = -3

    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, window: str | None = "hann",
                 keep_channels: bool = False, sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, keep_channels=keep_channels, sr=sr, **kwargs)
        _window(window, 1, torch.zeros(1))     # checks the window name
        self.window = window

    def windowed_frames(self, x: torch.Tensor) -> torch.Tensor:
        """(..., C, T) -> windowed frames (..., C, n_frames, frame_length)."""
        frames = _frames(x, self.frame_length, self.hop_length)
        window = _window(self.window, self.frame_length, frames)
        return frames if window is None else frames * window


@gin.configurable(module="features")
class DCT(_RepresentationFeature):
    """Discrete cosine transform (type II, orthonormal) of windowed frames : the n_coeffs first coefficients of
    scipy.fft.dct(frame, norm="ortho") (all of them if n_coeffs is None). Output (..., n_coeffs, n_frames).

    invert gives audio back by inverse DCT and weighted overlap-add of frames. Inversion is exact with all coefficients
    (n_coeffs=None) where windows overlap : with window=None (rectangular frames) and hop_length <= frame_length, or with
    hamming / hann windows that overlap (e.g. hop_length = frame_length // 2). With hann windows, the hop_length first
    and last samples of signals, only covered by vanishing window borders, are inaccurate (or lost). Samples after the
    last frame are lost (zeros with length). With fewer coefficients, inversion gives frames low-passed in the DCT
    domain."""
    invertible = True

    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, n_coeffs: int | None = None,
                 window: str | None = "hann", keep_channels: bool = False, sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, window=window, keep_channels=keep_channels, sr=sr, **kwargs)
        self.n_coeffs = int(n_coeffs or frame_length)
        assert self.n_coeffs <= self.frame_length, "n_coeffs must be <= frame_length"

    def __repr__(self):
        return "DCT(frame_length=%d, hop_length=%d, n_coeffs=%d, window=%s)"%(self.frame_length, self.hop_length, self.n_coeffs, self.window)

    @property
    def default_feature_name(self): return "dct"

    def compute(self, x):
        frames = self.windowed_frames(x)
        matrix = dct_matrix(self.n_coeffs, self.frame_length, dtype=frames.dtype, device=frames.device)
        return (frames @ matrix.T).transpose(-2, -1)

    def invert(self, features, length: int | None = None) -> torch.Tensor:
        """audio from DCT coefficients (..., [C], n_coeffs, n_frames) : (..., T) for the mix of channels, (..., C, T) with
        keep_channels, T = (n_frames - 1) * hop_length + frame_length (or length if given)."""
        y = _as_tensor(features)
        if not self.keep_channels:
            y = y.unsqueeze(-3)
        matrix = dct_matrix(self.n_coeffs, self.frame_length, dtype=y.dtype, device=y.device)
        frames = y.transpose(-2, -1) @ matrix                                    # windowed frames (..., C, n_frames, L)
        window = _window(self.window, self.frame_length, frames)
        if window is None:
            window = torch.ones(self.frame_length, dtype=frames.dtype, device=frames.device)
        # weighted overlap-add : least-squares estimate of the signal from its windowed frames
        x = _overlap_add(frames * window, self.hop_length)
        norm = _overlap_add(window.square().expand(frames.shape[-2], -1), self.hop_length)
        x = torch.where(norm > 1e-6, x / norm.clamp(min=1e-6), torch.zeros_like(x))
        if not self.keep_channels:
            x = x.squeeze(-2)
        return _fit_length(x, length)


def hz_to_erb_rate(f):
    """ERB-rate scale of Glasberg & Moore (1990) : number of ERBs below the frequency f (Hz)."""
    return 21.4 * torch.log10(1 + 0.00437 * f)


def erb_rate_to_hz(e):
    return (10 ** (e / 21.4) - 1) / 0.00437


def erb_bandwidth(f):
    """equivalent rectangular bandwidth (Hz) of auditory filters at the frequency f (Hz), Glasberg & Moore (1990)."""
    return 24.7 * (4.37 * f / 1000 + 1)


@gin.configurable(module="features")
class Gammatone(_RepresentationFeature):
    """Gammatone spectrogram : energy of frames in n_bands gammatone filters, with center frequencies equally spaced on
    the ERB-rate scale between fmin and fmax (Hz ; sr / 2 if None). Filters are applied in the frequency domain, on the
    power spectrum of windowed frames (n_fft = frame_length), with the magnitude response of gammatone filters of order
    filter_order (4) and bandwidth 1.019 ERB(fc) (Patterson et al., 1992), normalized to 1 at their center frequency.
    Output (..., n_bands, n_frames), in power (or dB if db). Center frequencies are given by center_frequencies."""

    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, n_bands: int = 64, fmin: float = 50.,
                 fmax: float | None = None, filter_order: int = 4, db: bool = False, window: str | None = "hann",
                 keep_channels: bool = False, sr: int = 44100, **kwargs):
        self.n_bands, self.fmin, self.filter_order, self.db = n_bands, fmin, filter_order, db
        super().__init__(frame_length=frame_length, hop_length=hop_length, window=window, keep_channels=keep_channels, sr=sr, **kwargs)
        self.fmax = fmax or sr / 2
        assert 0 <= self.fmin < self.fmax <= sr / 2, "fmin and fmax must verify 0 <= fmin < fmax <= sr / 2"
        erb_range = hz_to_erb_rate(torch.tensor([self.fmin, self.fmax], dtype=torch.float64))
        erb = torch.linspace(float(erb_range[0]), float(erb_range[1]), n_bands, dtype=torch.float64)
        self.center_frequencies = erb_rate_to_hz(erb)
        freqs = torch.fft.rfftfreq(frame_length, d=1. / sr).to(torch.float64)
        n = filter_order
        # bandwidth factor of gammatone filters of order n (1.019 for n = 4)
        factor = math.factorial(n - 1) ** 2 / (math.pi * math.factorial(2 * n - 2) * 2 ** -(2 * n - 2))
        bandwidth = factor * erb_bandwidth(self.center_frequencies)
        magnitude = (1 + ((freqs[None] - self.center_frequencies[:, None]) / bandwidth[:, None]) ** 2) ** (-n / 2)
        self.weights = (magnitude ** 2).float()       # (n_bands, K), applied on power spectra

    def __repr__(self):
        return "Gammatone(frame_length=%d, hop_length=%d, n_bands=%d, fmin=%s, fmax=%s)"%(self.frame_length, self.hop_length, self.n_bands, self.fmin, self.fmax)

    @property
    def default_feature_name(self): return "gammatone_%d"%self.n_bands

    def compute(self, x):
        frames = self.windowed_frames(x)
        power = torch.fft.rfft(frames, dim=-1).abs().square()
        bands = (power @ self.weights.to(power).T).transpose(-2, -1)
        return 10 * torch.log10(bands + _EPS) if self.db else bands


@gin.configurable(module="features")
class Cepstrum(_RepresentationFeature):
    """Real cepstrum of windowed frames : inverse Fourier transform of the log magnitude spectrum (n_fft = frame_length),
    c = irfft(log(|rfft(frame)| + eps)). The n_coeffs first coefficients are kept (frame_length // 2 + 1 if None : the
    cepstrum of real signals is symmetric). Coefficient q is the quefrency q / sr seconds (quefrencies attribute) : low
    coefficients describe the spectral envelope, and a periodic signal of fundamental frequency f0 has a peak at
    sr / f0. Output (..., n_coeffs, n_frames)."""

    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, n_coeffs: int | None = None,
                 window: str | None = "hann", keep_channels: bool = False, sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, window=window, keep_channels=keep_channels, sr=sr, **kwargs)
        self.n_coeffs = int(n_coeffs or frame_length // 2 + 1)
        assert self.n_coeffs <= self.frame_length, "n_coeffs must be <= frame_length"
        self.quefrencies = torch.arange(self.n_coeffs) / sr

    def __repr__(self):
        return "Cepstrum(frame_length=%d, hop_length=%d, n_coeffs=%d, window=%s)"%(self.frame_length, self.hop_length, self.n_coeffs, self.window)

    @property
    def default_feature_name(self): return "cepstrum"

    def compute(self, x):
        frames = self.windowed_frames(x)
        log_magnitude = torch.log(torch.fft.rfft(frames, dim=-1).abs() + _EPS)
        cepstrum = torch.fft.irfft(log_magnitude, n=self.frame_length, dim=-1)
        return cepstrum[..., :self.n_coeffs].transpose(-2, -1)


SPECTRUM_MODES = ["complex", "magnitude", "power", "db", "phase", "real", "imaginary", "power_density",
                  "instantaneous_phase", "instantaneous_frequency"]


def _parse_modes(mode, available):
    modes = [mode] if isinstance(mode, str) else list(mode)
    assert len(modes) > 0, "at least one mode must be given"
    for m in modes:
        assert m in available, "mode must be in %s, got %s"%(available, m)
    return modes


def _princarg(phase: torch.Tensor) -> torch.Tensor:
    """wraps phases in [-pi, pi[."""
    return torch.remainder(phase + math.pi, 2 * math.pi) - math.pi


def spectrum_mode(spectrum: torch.Tensor, mode: str, density_scale: torch.Tensor | None = None) -> torch.Tensor:
    """converts a complex spectrum to complex, magnitude, power, db (10 log10 power), phase, real, imaginary or
    power_density (power * density_scale) values."""
    if mode == "complex": return spectrum
    if mode == "magnitude": return spectrum.abs()
    if mode == "power": return spectrum.abs().square()
    if mode == "db": return 10 * torch.log10(spectrum.abs().square() + _EPS)
    if mode == "phase": return spectrum.angle()
    if mode == "real": return spectrum.real
    if mode == "imaginary": return spectrum.imag
    if mode == "power_density": return spectrum.abs().square() * density_scale
    raise ValueError("unknown mode %s"%mode)


def _density_scale(n_bins: int, n_fft: int, window: torch.Tensor | None, length: int, sr: int, onesided: bool, like: torch.Tensor):
    """scaling of power spectra of signals of length samples (windowed by window if not None) to power spectral
    densities (V^2 / Hz), as scipy.signal.periodogram(scaling="density")."""
    window_power = float(window.square().sum()) if window is not None else length
    scale = torch.full((n_bins,), 1. / (sr * window_power), dtype=like.dtype, device=like.device)
    if onesided:
        last = n_bins if n_fft % 2 else n_bins - 1     # Nyquist bin (even n_fft) is not doubled
        scale[1:last] *= 2
    return scale


_MAGNITUDE_MODES = ["magnitude", "power", "db", "power_density"]


def _split_modes(y: torch.Tensor, modes, keep_channels: bool, axis: int):
    """features (..., [n_modes * C], ...) with modes stacked on axis (modes first) -> {mode: (..., C, ...)}, with C = 1
    for mixes of channels. Real modes of complex outputs are given as real values."""
    n = len(modes)
    if n == 1 and not keep_channels:
        y = y.unsqueeze(axis)
    axis = axis % y.ndim
    shape = list(y.shape)
    if shape[axis] % n != 0:
        raise ValueError("features have %d channels, that are not a multiple of the number of modes (%d)"%(shape[axis], n))
    y = y.reshape(shape[:axis] + [n, shape[axis] // n] + shape[axis + 1:])
    values = {}
    for i, mode in enumerate(modes):
        value = y.select(axis, i)
        values[mode] = value if (mode == "complex" or not value.is_complex()) else value.real
    return values


def _magnitude_from_mode(value: torch.Tensor, mode: str, density_scale: torch.Tensor | None = None) -> torch.Tensor:
    if mode == "magnitude": return value.clamp(min=0)
    if mode == "power": return value.clamp(min=0).sqrt()
    if mode == "db": return (10 ** (value / 10) - _EPS).clamp(min=0).sqrt()
    if mode == "power_density": return (value / density_scale).clamp(min=0).sqrt()
    raise ValueError("unknown magnitude mode %s"%mode)


def spectrum_from_modes(values, density_scale: torch.Tensor | None = None):
    """inverse of spectrum_mode : (complex spectrum, None) from mode outputs {mode: values} that give the phase (complex,
    real and imaginary, or a magnitude and the phase), and (None, magnitude) from magnitudes only (magnitude, power, db,
    power_density). Raises a ValueError if modes give no magnitude."""
    if "complex" in values:
        return values["complex"], None
    if "real" in values and "imaginary" in values:
        return torch.complex(values["real"], values["imaginary"]), None
    magnitude = next((_magnitude_from_mode(values[m], m, density_scale) for m in _MAGNITUDE_MODES if m in values), None)
    if magnitude is None:
        raise ValueError("modes %s cannot be inverted : they must contain complex, (real, imaginary), or a magnitude "
                         "(magnitude, power, db, power_density)"%list(values.keys()))
    if "phase" in values:
        return torch.polar(magnitude, values["phase"]), None
    return None, magnitude


def _stack_modes(outputs):
    """stacks mode outputs on the channel axis (third last), as complex values if one of them is complex."""
    if any(o.is_complex() for o in outputs):
        outputs = [o if o.is_complex() else torch.complex(o, torch.zeros_like(o)) for o in outputs]
    return torch.cat(outputs, dim=-3) if len(outputs) > 1 else outputs[0]


class STFTModule(torch.nn.Module):
    """scriptable STFT of acids_transforms with the layout of acids_dataset features : (..., C, T) audio ->
    (..., K, n_frames) for the mix of channels, (..., C, K, n_frames) with keep_channels, and (..., n_modes * C, K,
    n_frames) for several modes (modes first) ; invert gives back audio (..., T) for the mix of channels, or (..., C, T).
    transform is an acids_transforms.STFT (offline) or StreamingSTFT (chunks of audio, see STFT.streaming)."""

    def __init__(self, transform: torch.nn.Module, keep_channels: bool, n_modes: int):
        super().__init__()
        self.transform = transform
        self.keep_channels = keep_channels
        self.n_modes = n_modes

    def to_feature(self, y: torch.Tensor) -> torch.Tensor:
        """(..., C, [n_modes], n_frames, K) -> feature layout (..., [n_modes * C], K, n_frames)."""
        y = y.transpose(-1, -2)
        if self.n_modes > 1:
            y = y.transpose(-4, -3)
            shape = list(y.shape)
            y = y.reshape(shape[:-4] + [shape[-4] * shape[-3], shape[-2], shape[-1]])
        elif not self.keep_channels:
            y = y.squeeze(-3)
        return y

    def from_feature(self, y: torch.Tensor) -> torch.Tensor:
        """feature layout (..., [n_modes * C], K, n_frames) -> (..., C, [n_modes], n_frames, K)."""
        if self.n_modes > 1:
            shape = list(y.shape)
            y = y.reshape(shape[:-3] + [self.n_modes, shape[-3] // self.n_modes, shape[-2], shape[-1]]).transpose(-4, -3)
        elif not self.keep_channels:
            y = y.unsqueeze(-3)
        return y.transpose(-1, -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if not self.keep_channels:
            x = x.mean(dim=-2, keepdim=True)
        return self.to_feature(self.transform(x))

    @torch.jit.export
    def invert(self, y: torch.Tensor, inversion_mode: Optional[str] = None) -> torch.Tensor:
        x = self.transform.invert(self.from_feature(y), inversion_mode)
        return x if self.keep_channels else x.squeeze(-2)


class STFTStreamingModule(STFTModule):
    """STFTModule of an acids_transforms.StreamingSTFT : forward transforms chunks of audio (..., C, T) (T a multiple
    of hop_length) into the frames they complete (..., K, T // hop_length), and invert gives back chunks of audio,
    delayed by latency samples. Call reset() between streams."""

    def __init__(self, transform: torch.nn.Module, keep_channels: bool, n_modes: int):
        super().__init__(transform, keep_channels, n_modes)
        self.latency: int = transform.latency
        self.hop_length: int = transform.hop_length

    @torch.jit.export
    def reset(self) -> None:
        self.transform.reset()


@gin.configurable(module="features")
class STFT(_RepresentationFeature):
    """Short-time Fourier transform, computed with acids_transforms.STFT : rfft of windowed frames of frame_length
    samples, zero-padded to n_fft (frame_length if None), every hop_length samples (frame_length // 4 if None). With
    center, signals are zero-padded with frame_length // 2 samples on both sides (frame t is centered on sample
    t * hop_length). Output (..., n_fft // 2 + 1, n_frames), or (..., C, n_fft // 2 + 1, n_frames) with keep_channels.

    mode is a mode, or a list of modes stacked on the channel axis ((..., n_modes * C, K, n_frames), modes first ; for
    the mix of channels, (..., n_modes, K, n_frames)). Outputs are real, unless "complex" is given (other modes are then
    cast to complex values). Modes :
        - "complex" : complex STFT.
        - "magnitude", "power", "db" (10 log10 power) : magnitude and power spectra.
        - "phase", "real", "imaginary" : phase (radians, in [-pi, pi]), real and imaginary parts.
        - "power_density" : power spectral density (V^2 / Hz) of each frame, as scipy.signal.spectrogram(
          scaling="density", mode="psd") : power / (sr * sum(window^2)), doubled for one-sided bins.
        - "instantaneous_phase" : phase difference between consecutive frames, wrapped in [-pi, pi] (time derivative of
          the unwrapped phase, as the instantaneous frequency of GANSynth up to a factor pi) ; 0 for the first frame.
        - "instantaneous_frequency" : instantaneous frequency of each bin in Hz, from the phase difference between
          frames minus the phase advance of the bin center (phase vocoder) : a stationary sinusoid of frequency f gives f
          in the bins of its peak. The first frame gives the center frequencies of bins.
    Frequencies of bins are given by the frequencies attribute.

    Features can be inverted (invert) : exactly from complex outputs, (real, imaginary), or a magnitude (magnitude,
    power, db, power_density) with the phase ; approximately with instantaneous phases or frequencies ; and with
    inversion_mode ("griffin_lim", "random", "keep_input", "sinebank") from magnitudes only. script() gives a
    TorchScript-compatible module (forward and invert), and streaming() a scriptable module computing the same features
    on chunks of audio, for real-time use (frames are the ones of center=False).
    """
    modes = SPECTRUM_MODES
    invertible = True

    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, n_fft: int | None = None,
                 mode: str | list = "magnitude", window: str | None = "hann", center: bool = False,
                 keep_channels: bool = False, sr: int = 44100, inversion_mode: str = "griffin_lim", **kwargs):
        self.mode = _parse_modes(mode, self.modes)
        super().__init__(frame_length=frame_length, hop_length=hop_length or frame_length // 4, window=window,
                         keep_channels=keep_channels, sr=sr, **kwargs)
        self.n_fft = int(n_fft or frame_length)
        assert self.n_fft >= self.frame_length, "n_fft must be >= frame_length"
        self.center = center
        assert inversion_mode in at.STFT.get_inversion_modes(), "inversion_mode must be in %s"%(at.STFT.get_inversion_modes(),)
        self.inversion_mode = inversion_mode
        self.frequencies = torch.fft.rfftfreq(self.n_fft, d=1. / sr)
        self._module = None

    def __repr__(self):
        return "STFT(frame_length=%d, hop_length=%d, n_fft=%d, mode=%s, window=%s, center=%s)"%(
            self.frame_length, self.hop_length, self.n_fft, self.mode, self.window, self.center)

    @property
    def default_feature_name(self):
        return "stft" if self.mode == ["magnitude"] else "stft_" + "_".join(self.mode)

    def _transform_kwargs(self):
        return dict(sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.frame_length,
                    window=self.window, mode=self.mode)

    def _build_module(self, streaming: bool = False) -> STFTModule:
        inversion_mode = getattr(self, "inversion_mode", "griffin_lim")
        if streaming:
            if inversion_mode not in at.StreamingSTFT.get_inversion_modes():
                inversion_mode = "random"
            transform = at.StreamingSTFT(inversion_mode=inversion_mode, **self._transform_kwargs())
            return STFTStreamingModule(transform, self.keep_channels, len(self.mode))
        transform = at.STFT(inversion_mode=inversion_mode, center=self.center, pad_mode="constant", **self._transform_kwargs())
        return STFTModule(transform, self.keep_channels, len(self.mode))

    def module(self, device: torch.device | None = None) -> STFTModule:
        """STFTModule computing the feature (built when needed : features pickled before acids_transforms have none)."""
        if getattr(self, "_module", None) is None:
            self._module = self._build_module()
        if device is not None and self._module.transform.window.device != device:
            self._module = self._module.to(device)
        return self._module

    def __call__(self, data) -> torch.Tensor:
        x = _as_tensor(data)
        return self.module(x.device)(x)

    def compute(self, x):
        """(..., C, T) -> (..., [n_modes * C], K, n_frames), keeping the channel axis."""
        return STFTModule(self.module(x.device).transform, True, len(self.mode))(x)

    def invert(self, features, inversion_mode: str | None = None, length: int | None = None) -> torch.Tensor:
        """audio from features (outputs of the feature) : (..., T) for the mix of channels, (..., C, T) with
        keep_channels, cropped or zero-padded to length samples if given. inversion_mode (see acids_transforms.STFT) is
        used for features without phase."""
        y = torch.as_tensor(features)
        return _fit_length(self.module(y.device).invert(y, inversion_mode), length)

    def script(self) -> STFTModule:
        """TorchScript-compatible module (torch.jit.script) computing the feature (forward) and inverting it (invert)."""
        return self._build_module()

    def streaming(self) -> STFTStreamingModule:
        """TorchScript-compatible module computing the feature on chunks of audio (..., C, T), T a multiple of
        hop_length, for real-time use : each chunk gives T // hop_length frames, that are the ones of the feature with
        center=False, preceded by frame_length // hop_length - 1 frames that contain the initial zeros of the buffer.
        invert gives back chunks of audio, delayed by latency (frame_length - hop_length) samples. Inversion modes are
        "random", "keep_input" and "sinebank" (inversion_mode of the feature if available, random otherwise)."""
        if self.center:
            logging.warning("streaming STFT gives frames of center=False : frames of %s are shifted by frame_length // 2 samples"%self)
        return self._build_module(streaming=True)


@gin.configurable(module="features")
class DGT(_RepresentationFeature):
    """Discrete Gabor transform, computed with acids_transforms.DGT : STFT with a gaussian window of n_fft samples, every
    hop_length samples (n_fft // 4 if None). Frames are centered (signals are padded by reflection with n_fft // 2
    samples on both sides), such that frame t is centered on sample t * hop_length : signals of T samples give
    T // hop_length + 1 frames. Output (..., n_fft // 2 + 1, n_frames), or (..., C, n_fft // 2 + 1, n_frames) with
    keep_channels.

    mode is a mode, or a list of modes stacked on the channel axis (modes first), as in STFT : "complex", "magnitude",
    "power", "db", "phase", "real" and "imaginary". Outputs are real, unless "complex" is given.

    invert gives audio back : exactly from complex outputs, (real, imaginary), or a magnitude (magnitude, power, db)
    with the phase ; from magnitudes only, with inversion_mode ("pghi" by default : phase gradient heuristic integration,
    well suited to gaussian windows ; "griffin_lim", "random", "keep_input", "sinebank"). Inverted signals have
    (n_frames - 1) * hop_length samples (or length if given).
    """
    modes = ["complex", "magnitude", "power", "db", "phase", "real", "imaginary"]
    invertible = True

    def __init__(self, n_fft: int = 1024, hop_length: int | None = None, mode: str | list = "magnitude",
                 keep_channels: bool = False, sr: int = 44100, inversion_mode: str = "pghi", tolerance: float = 1e-2, **kwargs):
        self.mode = _parse_modes(mode, self.modes)
        super().__init__(frame_length=n_fft, hop_length=hop_length or n_fft // 4, window=None,
                         keep_channels=keep_channels, sr=sr, **kwargs)
        self.n_fft = int(n_fft)
        assert inversion_mode in at.DGT.get_inversion_modes(), "inversion_mode must be in %s"%(at.DGT.get_inversion_modes(),)
        self.inversion_mode = inversion_mode
        self.tolerance = tolerance
        self.frequencies = torch.fft.rfftfreq(self.n_fft, d=1. / sr)
        self._module = None

    def __repr__(self):
        return "DGT(n_fft=%d, hop_length=%d, mode=%s, inversion_mode=%s)"%(self.n_fft, self.hop_length, self.mode, self.inversion_mode)

    @property
    def default_feature_name(self):
        return "dgt" if self.mode == ["magnitude"] else "dgt_" + "_".join(self.mode)

    def module(self, device: torch.device | None = None) -> torch.nn.Module:
        """acids_transforms.DGT computing the feature (built when needed)."""
        if getattr(self, "_module", None) is None:
            self._module = at.DGT(sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
                                  inversion_mode=self.inversion_mode, tolerance=self.tolerance)
        if device is not None and self._module.window.device != device:
            self._module = self._module.to(device)
        return self._module

    def compute(self, x):
        """(..., C, T) -> (..., [n_modes * C], K, n_frames), keeping the channel axis."""
        spectrum = self.module(x.device)(x).transpose(-1, -2)                  # (..., C, K, n_frames)
        return _stack_modes([spectrum_mode(spectrum, mode) for mode in self.mode])

    def invert(self, features, inversion_mode: str | None = None, length: int | None = None) -> torch.Tensor:
        """audio from features (outputs of the feature) : (..., T) for the mix of channels, (..., C, T) with
        keep_channels, cropped or zero-padded to length samples if given. inversion_mode (see acids_transforms.DGT) is
        used for features without phase."""
        y = torch.as_tensor(features)
        module = self.module(y.device)
        spectrum, magnitude = spectrum_from_modes(_split_modes(y, self.mode, self.keep_channels, axis=-3))
        if spectrum is not None:
            x = module.invert(spectrum.transpose(-1, -2))
        else:
            x = module.invert(magnitude.transpose(-1, -2).contiguous(), inversion_mode or self.inversion_mode)
        if not self.keep_channels:
            x = x.squeeze(-2)
        return _fit_length(x, length)


@gin.configurable(module="features")
class FFT(_WaveformFeature):
    """Fourier transform of whole signals (chunks) : (..., C, T) -> (..., K), or (..., C, K) with keep_channels.
    Signals are cropped to n_fft samples (or zero-padded, after windowing) if n_fft is given, and windowed (window, None
    by default). With onesided (default), K = n_fft // 2 + 1 non-negative frequencies (rfft) ; otherwise, the full
    spectrum of the n_fft frequencies (fft, in the order of torch.fft.fftfreq).

    mode is a mode, or a list of modes stacked on the channel axis ((..., n_modes * C, K), modes first ; for the mix of
    channels, (..., n_modes, K)) : "complex", "magnitude", "power", "db", "phase", "real", "imaginary" and
    "power_density" (V^2 / Hz, as scipy.signal.periodogram(scaling="density")), as in STFT. Outputs are real, unless
    "complex" is given. Frequencies of bins are given by frequencies(n_samples).

    invert gives signals back exactly from complex outputs, (real, imaginary), or a magnitude (magnitude, power, db,
    power_density) with the phase. Windows are divided out where they do not vanish (window=None or "hamming" for exact
    inversions ; hann windows lose the first sample). Samples cropped by n_fft are lost (zeros with length).
    """
    modes = SPECTRUM_MODES[:-2]
    invertible = True

    def __init__(self, n_fft: int | None = None, mode: str | list = "magnitude", window: str | None = None,
                 onesided: bool = True, keep_channels: bool = False, sr: int = 44100, **kwargs):
        self.mode = _parse_modes(mode, self.modes)
        _window(window, 1, torch.zeros(1))     # checks the window name
        self.n_fft, self.window, self.onesided, self.keep_channels = n_fft, window, onesided, keep_channels
        super().__init__(sr=sr, **kwargs)

    def __repr__(self):
        return "FFT(n_fft=%s, mode=%s, window=%s, onesided=%s)"%(self.n_fft, self.mode, self.window, self.onesided)

    @property
    def default_feature_name(self):
        return "fft" if self.mode == ["magnitude"] else "fft_" + "_".join(self.mode)

    def frequencies(self, n_samples: int) -> torch.Tensor:
        """frequencies (Hz) of the bins of signals of n_samples samples."""
        n = self.n_fft or n_samples
        return torch.fft.rfftfreq(n, d=1. / self.sr) if self.onesided else torch.fft.fftfreq(n, d=1. / self.sr)

    def __call__(self, data) -> torch.Tensor:
        x = _as_tensor(data)
        if not self.keep_channels:
            x = x.mean(dim=-2, keepdim=True)
        n = self.n_fft or x.shape[-1]
        x = x[..., :n]                          # cropped to n_fft samples (zero-padded by the FFT otherwise)
        window = _window(self.window, x.shape[-1], x)
        if window is not None:
            x = x * window
        spectrum = torch.fft.rfft(x, n=n, dim=-1) if self.onesided else torch.fft.fft(x, n=n, dim=-1)   # (..., C, K)
        scale = None
        if "power_density" in self.mode:
            scale = _density_scale(spectrum.shape[-1], n, window, x.shape[-1], self.sr, self.onesided, x)
        outputs = [spectrum_mode(spectrum, mode, scale)[..., None, :, :] for mode in self.mode]  # (..., 1, C, K)
        out = _stack_modes(outputs).flatten(-3, -2) if len(outputs) > 1 else outputs[0][..., 0, :, :]
        return out if (self.keep_channels or len(self.mode) > 1) else out.squeeze(-2)

    def invert(self, features, length: int | None = None) -> torch.Tensor:
        """signals from FFT features (..., [n_modes * C], K) : (..., T) for the mix of channels, (..., C, T) with
        keep_channels. T is n_fft if given, length otherwise (n_fft = 2 * (K - 1) for one-sided spectra without length :
        give length for signals of odd length). Raises a ValueError for features without phase."""
        y = torch.as_tensor(features)
        n_bins = y.shape[-1]
        n = self.n_fft or length or (2 * (n_bins - 1) if self.onesided else n_bins)
        n_samples = min(length, n) if length is not None else n       # samples of the signal analyzed by the FFT
        values = _split_modes(y, self.mode, self.keep_channels, axis=-2)
        like = next(iter(values.values()))
        window = _window(self.window, n_samples, like.real if like.is_complex() else like)
        scale = None
        if "power_density" in self.mode:
            scale = _density_scale(n_bins, n, window, n_samples, self.sr, self.onesided, like.real if like.is_complex() else like)
        spectrum, _ = spectrum_from_modes(values, scale)
        if spectrum is None:
            raise ValueError("FFT features with modes %s have no phase, and cannot be inverted"%self.mode)
        x = torch.fft.irfft(spectrum, n=n, dim=-1) if self.onesided else torch.fft.ifft(spectrum, n=n, dim=-1).real
        x = x[..., :n_samples]
        if window is not None:
            x = torch.where(window > 1e-6, x / window.clamp(min=1e-6), torch.zeros_like(x))
        if not self.keep_channels:
            x = x.squeeze(-2)
        return _fit_length(x, length)


__all__ = ['STFT', 'STFTModule', 'STFTStreamingModule', 'DGT', 'FFT', 'DCT', 'Gammatone', 'Cepstrum', 'spectrum_mode', 'spectrum_from_modes', 'SPECTRUM_MODES']
