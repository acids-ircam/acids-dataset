"""Linear predictive coding (LPC) per frame : prediction coefficients, reflection coefficients, line spectral
frequencies and spectral envelopes, computed with the autocorrelation method and the Levinson-Durbin recursion."""
import math
import gin
import torch

from .framed import _EPS, _frames, _window, _ChannelFeature


def levinson_durbin(r: torch.Tensor, order: int):
    """Levinson-Durbin recursion on autocorrelations r (..., >= order + 1). Returns the prediction polynomial
    a (..., order + 1) with a[..., 0] = 1 (A(z) = 1 + a_1 z^-1 + ... + a_p z^-p, x[n] ~ -sum_k a_k x[n - k]), the
    reflection coefficients k (..., order), and the prediction error energy (...). Silent frames give A(z) = 1, k = 0
    and an error of 0."""
    a = torch.zeros(r.shape[:-1] + (order + 1,), dtype=r.dtype, device=r.device)
    a[..., 0] = 1.
    error = r[..., 0].clone()
    reflections = []
    for i in range(1, order + 1):
        acc = (a[..., :i] * r[..., 1:i + 1].flip(-1)).sum(-1)
        k = torch.where(error > _EPS, -acc / error.clamp(min=_EPS), torch.zeros_like(acc))
        a_prev = a.clone()
        a[..., 1:i] = a_prev[..., 1:i] + k[..., None] * a_prev[..., 1:i].flip(-1)
        a[..., i] = k
        error = error * (1 - k ** 2)
        reflections.append(k)
    return a, torch.stack(reflections, dim=-1), error.clamp(min=0)


def _deflate(c: torch.Tensor, root: float) -> torch.Tensor:
    """divides polynomials in z^-1 (coefficients c (..., n + 1)) by (1 - root z^-1), for an exact root."""
    q = [c[..., 0]]
    for k in range(1, c.shape[-1] - 1):
        q.append(c[..., k] + root * q[-1])
    return torch.stack(q, dim=-1)


def _root_angles(c: torch.Tensor) -> torch.Tensor:
    """sorted angles in [0, pi] of the roots (in conjugate pairs) of polynomials in z^-1 (..., n + 1) : (..., n // 2)."""
    n = c.shape[-1] - 1
    companion = torch.zeros(c.shape[:-1] + (n, n), dtype=torch.float64, device=c.device)
    companion[..., 0, :] = -c[..., 1:].double() / c[..., :1].double()
    companion[..., 1:, :-1] = torch.eye(n - 1, dtype=torch.float64, device=c.device)
    roots = torch.linalg.eigvals(companion)
    angles = roots.angle().abs().sort(dim=-1).values
    return angles[..., ::2].to(c.dtype)


def line_spectral_frequencies(a: torch.Tensor) -> torch.Tensor:
    """line spectral frequencies (radians, in ]0, pi[, increasing) of prediction polynomials a (..., p + 1) : the angles
    of the roots of P(z) = A(z) + z^-(p+1) A(1/z) and Q(z) = A(z) - z^-(p+1) A(1/z), without the trivial roots +/- 1.
    Output (..., p)."""
    p = a.shape[-1] - 1
    extended = torch.nn.functional.pad(a, (0, 1))
    P, Q = extended + extended.flip(-1), extended - extended.flip(-1)
    if p % 2 == 0:
        P, Q = _deflate(P, -1.), _deflate(Q, 1.)
    else:
        Q = _deflate(_deflate(Q, 1.), -1.)
    lsf = torch.cat([_root_angles(P), _root_angles(Q)], dim=-1)
    return lsf.sort(dim=-1).values


@gin.configurable(module="features")
class LPC(_ChannelFeature):
    """Linear predictive coding of windowed frames (autocorrelation method, Levinson-Durbin recursion), with a
    prediction polynomial A(z) = 1 + a_1 z^-1 + ... + a_p z^-p of order p (x[n] ~ -sum_k a_k x[n - k]).

    Modes :
        - "lpc" (default) : prediction coefficients a_1 ... a_p, (..., order, n_frames). With the leading 1, they are
          the denominator of the all-pole synthesis filter (e.g. scipy.signal.lfilter([gain], [1, *a], excitation)).
        - "reflection" : reflection (PARCOR) coefficients k_1 ... k_p, in ]-1, 1[, (..., order, n_frames).
        - "lsf" : line spectral frequencies, in radians (]0, pi[, increasing), (..., order, n_frames).
        - "envelope" : LPC spectral envelope gain^2 / |A(e^jw)|^2 in dB, on the frame_length // 2 + 1 bins of the
          power spectrum of the windowed frame, (..., frame_length // 2 + 1, n_frames).
    With gain=True, the gain (square root of the prediction error energy of the frame) is prepended to the "lpc",
    "reflection" and "lsf" outputs ((..., order + 1, n_frames)). Signals are pre-emphasized with
    x[n] - pre_emphasis * x[n - 1] (0 : no pre-emphasis ; 0.97 is common for speech). Silent frames give A(z) = 1.
    """
    modes = ["lpc", "reflection", "lsf", "envelope"]
    _channel_axis = -3

    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, order: int = 16, mode: str = "lpc",
                 gain: bool = False, pre_emphasis: float = 0., window: str | None = "hann", keep_channels: bool = False,
                 sr: int = 44100, **kwargs):
        assert mode in self.modes, "mode must be in %s, got %s"%(self.modes, mode)
        assert 0 < order < frame_length, "order must be in [1, frame_length["
        _window(window, 1, torch.zeros(1))     # checks the window name
        self.order, self.mode, self.gain, self.pre_emphasis, self.window = order, mode, gain, pre_emphasis, window
        super().__init__(frame_length=frame_length, hop_length=hop_length, keep_channels=keep_channels, sr=sr, **kwargs)

    def __repr__(self):
        return "LPC(frame_length=%d, hop_length=%d, order=%d, mode=%s, gain=%s)"%(self.frame_length, self.hop_length, self.order, self.mode, self.gain)

    @property
    def default_feature_name(self):
        return "lpc" if self.mode == "lpc" else "lpc_%s"%self.mode

    def analyze(self, x: torch.Tensor):
        """(..., C, T) -> prediction polynomials (..., C, n_frames, order + 1), reflection coefficients
        (..., C, n_frames, order) and prediction error energies (..., C, n_frames)."""
        if self.pre_emphasis:
            x = torch.cat([x[..., :1], x[..., 1:] - self.pre_emphasis * x[..., :-1]], dim=-1)
        frames = _frames(x, self.frame_length, self.hop_length)
        window = _window(self.window, self.frame_length, frames)
        if window is not None:
            frames = frames * window
        # autocorrelation (biased) through the FFT, zero-padded to avoid circular aliasing
        n_fft = 2 ** math.ceil(math.log2(2 * self.frame_length))
        spectrum = torch.fft.rfft(frames.double(), n=n_fft, dim=-1)
        r = torch.fft.irfft(spectrum.abs().square(), n=n_fft, dim=-1)[..., :self.order + 1]
        a, k, error = levinson_durbin(r, self.order)
        return a.to(x.dtype), k.to(x.dtype), error.to(x.dtype)

    def compute(self, x):
        a, k, error = self.analyze(x)
        if self.mode == "envelope":
            response = torch.fft.rfft(a, n=self.frame_length, dim=-1).abs().square()
            envelope = 10 * torch.log10(error[..., None] + _EPS) - 10 * torch.log10(response + _EPS)
            return envelope.transpose(-2, -1)
        if self.mode == "lpc":
            out = a[..., 1:]
        elif self.mode == "reflection":
            out = k
        else:
            out = line_spectral_frequencies(a)
        if self.gain:
            out = torch.cat([error.sqrt()[..., None], out], dim=-1)
        return out.transpose(-2, -1)


__all__ = ['LPC', 'levinson_durbin', 'line_spectral_frequencies']
