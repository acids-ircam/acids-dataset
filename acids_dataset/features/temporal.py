"""Temporal descriptors : RMS energy and zero-crossing rate per frame, and envelope descriptors per chunk (attack time,
temporal centroid, crest factor)."""
import gin
import torch

from .framed import _EPS, _frames, _ChannelFeature


@gin.configurable(module="features")
class RMS(_ChannelFeature):
    """root mean square energy per frame, in dB (floored to min_db) or linear (db=False)."""
    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, db: bool = True, min_db: float = -100.,
                 keep_channels: bool = False, sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, keep_channels=keep_channels, sr=sr, **kwargs)
        self.db, self.min_db = db, min_db

    @property
    def default_feature_name(self): return "rms"

    def compute(self, x):
        rms = torch.sqrt(_frames(x, self.frame_length, self.hop_length).square().mean(-1))
        return (20 * torch.log10(rms + _EPS)).clamp(min=self.min_db) if self.db else rms


@gin.configurable(module="features")
class ZeroCrossingRate(_ChannelFeature):
    """sign changes per frame, as a ratio of samples (unit="ratio"), or as a frequency (unit="hz" : the frequency of a sine
    with the same rate)."""
    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, unit: str = "ratio", keep_channels: bool = False,
                 sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, keep_channels=keep_channels, sr=sr, **kwargs)
        assert unit in ["ratio", "hz"], "unit must be ratio or hz"
        self.unit = unit

    @property
    def default_feature_name(self): return "zero_crossing_rate"

    def compute(self, x):
        frames = _frames(x, self.frame_length, self.hop_length)
        signs = torch.sign(frames)
        crossings = (signs[..., 1:] * signs[..., :-1] < 0).float().mean(-1)
        return crossings * self.sr / 2 if self.unit == "hz" else crossings


class _EnvelopeFeature(_ChannelFeature):
    """descriptors of the RMS envelope of a whole chunk (envelope frames of envelope_length samples), output (...,) or
    (..., C) with keep_channels."""
    _channel_axis = -1

    def __init__(self, envelope_length: int = 256, keep_channels: bool = False, sr: int = 44100, **kwargs):
        super().__init__(frame_length=envelope_length, hop_length=envelope_length, keep_channels=keep_channels, sr=sr, **kwargs)

    def envelope(self, x):
        return torch.sqrt(_frames(x, self.frame_length, self.hop_length).square().mean(-1))


@gin.configurable(module="features")
class AttackTime(_EnvelopeFeature):
    """time (seconds) for the envelope of a chunk to rise from low (10%) to high (90%) of its maximum, before the maximum."""
    def __init__(self, envelope_length: int = 256, low: float = 0.1, high: float = 0.9, keep_channels: bool = False, sr: int = 44100, **kwargs):
        super().__init__(envelope_length=envelope_length, keep_channels=keep_channels, sr=sr, **kwargs)
        self.low, self.high = low, high

    @property
    def default_feature_name(self): return "attack_time"

    def compute(self, x):
        env = self.envelope(x)
        peak_value, peak_index = env.max(-1)
        positions = torch.arange(env.shape[-1], device=env.device)
        before_peak = positions <= peak_index[..., None]
        # last frame below each threshold before the peak
        below_low = (env < self.low * peak_value[..., None]) & before_peak
        below_high = (env < self.high * peak_value[..., None]) & before_peak
        start = torch.where(below_low.any(-1), (below_low * positions).max(-1).values, torch.zeros_like(peak_index))
        end = torch.where(below_high.any(-1), (below_high * positions).max(-1).values + 1, torch.zeros_like(peak_index))
        attack = (end - start).clamp(min=0).float() * self.hop_length / self.sr
        return torch.where(peak_value > _EPS, attack, torch.zeros_like(attack))


@gin.configurable(module="features")
class TemporalCentroid(_EnvelopeFeature):
    """center of mass of the envelope of a chunk, in seconds."""
    @property
    def default_feature_name(self): return "temporal_centroid"

    def compute(self, x):
        env = self.envelope(x)
        times = (torch.arange(env.shape[-1], device=env.device, dtype=env.dtype) + 0.5) * self.hop_length / self.sr
        total = env.sum(-1)
        return torch.where(total > _EPS, (env * times).sum(-1) / total.clamp(min=_EPS), torch.zeros_like(total))


@gin.configurable(module="features")
class CrestFactor(_ChannelFeature):
    """peak / RMS ratio of a chunk, in dB (db=True) or linear."""
    _channel_axis = -1

    def __init__(self, db: bool = True, keep_channels: bool = False, sr: int = 44100, **kwargs):
        super().__init__(frame_length=1, keep_channels=keep_channels, sr=sr, **kwargs)
        self.db = db

    @property
    def default_feature_name(self): return "crest_factor"

    def compute(self, x):
        peak, rms = x.abs().amax(-1), torch.sqrt(x.square().mean(-1))
        crest = torch.where(rms > _EPS, peak / rms.clamp(min=_EPS), torch.zeros_like(rms))
        return 20 * torch.log10(crest + _EPS) if self.db else crest


__all__ = ['RMS', 'ZeroCrossingRate', 'AttackTime', 'TemporalCentroid', 'CrestFactor']
