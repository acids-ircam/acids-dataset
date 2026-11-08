import gin
import math
import torch
import torchaudio
from typing import List, Optional

from . import RAISE_EXC_IF_FEATURE_ERROR
from .base import AcidsDatasetFeature
from ..utils import loudness, frame_with_pad


@gin.configurable(module="features")
class Loudness(AcidsDatasetFeature):
    supports_batch = True
    def __init__(
            self, 
            frame_length: int | None = None,
            keep_channels: bool = True,
            sr: int = 44100, 
            min_loudness: float | None = -70.,
            **kwargs
    ):
        """Loudness (LUFS), per frame if frame_length is given. Values are floored to min_loudness (by default -70 LUFS, 
        the absolute gate of loudness measurements), such that silence does not give -inf ; None keeps -inf for silence."""
        super().__init__(**kwargs)
        self.min_loudness = min_loudness
        self.frame_length = frame_length
        self.sr = sr
        self.keep_channels = keep_channels
        self.kwargs = kwargs

    def __repr__(self):
        return "Loudness(sr=%d)"%(self.sr)

    @property
    def has_hash(self):
        return False

    @property
    def default_feature_name(self):
        return "loudness"

    def from_fragment(self, fragment, write: bool = True):
        data = torch.from_numpy(fragment.raw_audio).float()
        try: 
            data_loudness = self(data)
        except RuntimeError as e:
            if RAISE_EXC_IF_FEATURE_ERROR: raise e
            return  
        if write:
            fragment.put_array(self.feature_name, data_loudness)

    def __call__(self, data) -> torch.Tensor:
        data_loudness = loudness(data, sample_rate=self.sr, frame_length=self.frame_length, keep_channels=self.keep_channels)
        # features pickled before min_loudness have no floor
        min_loudness = getattr(self, "min_loudness", None)
        if min_loudness is not None:
            data_loudness = data_loudness.clamp(min=min_loudness)
        return data_loudness

    def script(self):
        return LoudnessScript(frame_length=self.frame_length, sr=self.sr, keep_channels=self.keep_channels, min_loudness=getattr(self, "min_loudness", None))


class LoudnessScript(torch.nn.Module):
    """TorchScript-compatible loudness module (pure-PyTorch K-weighting, no torchaudio dependency)."""

    frame_length: Optional[int]
    sr: int
    keep_channels: bool
    min_loudness: Optional[float]
    hs_b0: float
    hs_b1: float
    hs_b2: float
    hs_a1: float
    hs_a2: float
    hp_b0: float
    hp_b1: float
    hp_b2: float
    hp_a1: float
    hp_a2: float

    def __init__(
        self,
        frame_length: int | None = None,
        keep_channels: bool = True,
        sr: int = 44100,
        min_loudness: Optional[float] = -70.,
    ):
        super().__init__()
        self.min_loudness = min_loudness
        self.frame_length = frame_length
        self.sr = sr
        self.keep_channels = keep_channels

        # K-weighting stage 1: treble shelf (gain=4 dB, fc=1500 Hz, Q=1/sqrt(2))
        gain, cf, Q = 4.0, 1500.0, 1.0 / math.sqrt(2.0)
        w0 = 2.0 * math.pi * cf / sr
        A = 10.0 ** (gain / 40.0)
        alpha = math.sin(w0) / (2.0 * Q)
        b0 =  A * ((A + 1) + (A - 1) * math.cos(w0) + 2.0 * math.sqrt(A) * alpha)
        b1 = -2.0 * A * ((A - 1) + (A + 1) * math.cos(w0))
        b2 =  A * ((A + 1) + (A - 1) * math.cos(w0) - 2.0 * math.sqrt(A) * alpha)
        a0 =      (A + 1) - (A - 1) * math.cos(w0) + 2.0 * math.sqrt(A) * alpha
        a1 =  2.0 * ((A - 1) - (A + 1) * math.cos(w0))
        a2 =      (A + 1) - (A - 1) * math.cos(w0) - 2.0 * math.sqrt(A) * alpha
        self.hs_b0, self.hs_b1, self.hs_b2 = b0 / a0, b1 / a0, b2 / a0
        self.hs_a1, self.hs_a2 = a1 / a0, a2 / a0

        # K-weighting stage 2: high-pass (fc=38 Hz, Q=0.5)
        cf2, Q2 = 38.0, 0.5
        w0 = 2.0 * math.pi * cf2 / sr
        alpha = math.sin(w0) / (2.0 * Q2)
        b0 =  (1.0 + math.cos(w0)) / 2.0
        b1 = -(1.0 + math.cos(w0))
        b2 =  (1.0 + math.cos(w0)) / 2.0
        a0 =  1.0 + alpha
        a1 = -2.0 * math.cos(w0)
        a2 =  1.0 - alpha
        self.hp_b0, self.hp_b1, self.hp_b2 = b0 / a0, b1 / a0, b2 / a0
        self.hp_a1, self.hp_a2 = a1 / a0, a2 / a0

    def _biquad(self, x: torch.Tensor,
                b0: float, b1: float, b2: float,
                a1: float, a2: float) -> torch.Tensor:
        """Direct Form II Transposed biquad: no torchaudio/_lfilter dependency."""
        T = x.shape[-1]
        y = torch.empty_like(x)
        s1 = torch.zeros_like(x[..., 0])
        s2 = torch.zeros_like(x[..., 0])
        for t in range(T):
            xt = x[..., t]
            yt = b0 * xt + s1
            s1 = b1 * xt - a1 * yt + s2
            s2 = b2 * xt - a2 * yt
            y[..., t] = yt
        return y

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        C = waveform.size(-2)
        if C > 5:
            raise ValueError("Only up to 5 channels are supported.")

        batch_shape = list(waveform.shape[:-2])
        T = waveform.size(-1)

        B = 1
        for s in batch_shape:
            B *= s

        num_frames: int | None = None

        frame_length = self.frame_length
        if frame_length is not None:
            remainder = T % frame_length
            if remainder != 0:
                waveform = waveform[..., :frame_length * (T // frame_length)]
            waveform = waveform.unfold(-1, frame_length, frame_length)
            num_frames = waveform.size(-2)
            ndim = waveform.ndim
            perm = list(range(ndim - 3)) + [ndim - 2, ndim - 3, ndim - 1]
            waveform = waveform.permute(perm).contiguous()
            waveform = waveform.reshape(B * num_frames, C, frame_length)
        else:
            waveform = waveform.reshape(B, C, T)

        waveform = self._biquad(waveform, self.hs_b0, self.hs_b1, self.hs_b2, self.hs_a1, self.hs_a2)
        waveform = self._biquad(waveform, self.hp_b0, self.hp_b1, self.hp_b2, self.hp_a1, self.hp_a2)

        T_filt = waveform.size(-1)
        gate_duration = min(0.4, T_filt / self.sr)
        kweight_bias = -0.691
        gamma_abs = -70.0
        gate_samples = int(round(gate_duration * self.sr))
        step = int(round(gate_samples * 0.25))

        energy = torch.square(waveform).unfold(-1, gate_samples, step).mean(dim=-1)
        g = torch.tensor([1.0, 1.0, 1.0, 1.41, 1.41], dtype=waveform.dtype, device=waveform.device)[:C]

        if self.keep_channels:
            lval = kweight_bias + 10.0 * torch.log10(energy.clamp(min=1e-10))
            gated = lval > gamma_abs
            count = gated.sum(dim=-1).clamp(min=1).float()
            energy_filtered = (gated.float() * energy).sum(dim=-1) / count
            gamma_rel = kweight_bias + 10.0 * torch.log10(energy_filtered.clamp(min=1e-10)) - 10.0

            gated_rel = gated & (lval > gamma_rel.unsqueeze(-1))
            count_rel = gated_rel.sum(dim=-1).clamp(min=1).float()
            energy_final = (gated_rel.float() * energy).sum(dim=-1) / count_rel

            LKFS = torch.where(
                gated_rel.any(dim=-1),
                kweight_bias + 10.0 * torch.log10(energy_final.clamp(min=1e-10)),
                torch.full_like(energy_final, float('-inf')),
            )

            if num_frames is not None:
                LKFS = LKFS.reshape(batch_shape + [num_frames, C]).transpose(-2, -1)
            else:
                LKFS = LKFS.reshape(batch_shape + [C])
        else:
            energy_w = torch.einsum('bcw,c->bw', energy, g)
            lval = kweight_bias + 10.0 * torch.log10(energy_w.clamp(min=1e-10))

            gated = lval > gamma_abs
            count = gated.sum(dim=-1).clamp(min=1).float()
            energy_filtered = torch.einsum('bcw,bw->bc', energy, gated.float()) / count.unsqueeze(-1)
            energy_w_f = torch.einsum('bc,c->b', energy_filtered, g)
            gamma_rel = kweight_bias + 10.0 * torch.log10(energy_w_f.clamp(min=1e-10)) - 10.0

            gated_rel = gated & (lval > gamma_rel.unsqueeze(-1))
            count_rel = gated_rel.sum(dim=-1).clamp(min=1).float()
            energy_filtered2 = torch.einsum('bcw,bw->bc', energy, gated_rel.float()) / count_rel.unsqueeze(-1)
            energy_w_f2 = torch.einsum('bc,c->b', energy_filtered2, g)

            LKFS = torch.where(
                gated_rel.any(dim=-1),
                kweight_bias + 10.0 * torch.log10(energy_w_f2.clamp(min=1e-10)),
                torch.full_like(energy_w_f2, float('-inf')),
            )

            if len(batch_shape) > 0:
                if num_frames is not None:
                    LKFS = LKFS.reshape(batch_shape + [num_frames])
                else:
                    LKFS = LKFS.reshape(batch_shape)
            else:
                LKFS = LKFS.squeeze(0)

        min_loudness = self.min_loudness
        if min_loudness is not None:
            LKFS = LKFS.clamp(min=min_loudness)
        return LKFS
