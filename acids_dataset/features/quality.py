"""Quality and dynamics descriptors per chunk : clipping ratio, DC offset, dynamic range and silence ratio (see also
acids_dataset.health for reports on whole datasets)."""
import gin
import torch

from .framed import _EPS, _frames, _ChannelFeature


class _ChunkFeature(_ChannelFeature):
    """descriptors of whole chunks : output (...,), or (..., C) with keep_channels."""
    _channel_axis = -1


@gin.configurable(module="features")
class ClippingRatio(_ChunkFeature):
    """ratio of samples whose absolute value reaches threshold (clipped samples)."""
    def __init__(self, threshold: float = 0.999, keep_channels: bool = False, sr: int = 44100, **kwargs):
        super().__init__(frame_length=1, keep_channels=keep_channels, sr=sr, **kwargs)
        self.threshold = threshold

    @property
    def default_feature_name(self): return "clipping_ratio"

    def _channels(self, data):
        # clipping is measured on channels (mixes would hide it), then averaged if keep_channels is False
        from .framed import _as_tensor
        return _as_tensor(data)

    def compute(self, x):
        ratio = (x.abs() >= self.threshold).float().mean(-1)
        return ratio if self.keep_channels else ratio.mean(-1, keepdim=True)


@gin.configurable(module="features")
class DCOffset(_ChunkFeature):
    """mean value of the signal (of each channel with keep_channels, of the mix otherwise)."""
    def __init__(self, keep_channels: bool = True, sr: int = 44100, **kwargs):
        super().__init__(frame_length=1, keep_channels=keep_channels, sr=sr, **kwargs)

    @property
    def default_feature_name(self): return "dc_offset"

    def compute(self, x):
        return x.mean(-1)


@gin.configurable(module="features")
class DynamicRange(_ChunkFeature):
    """difference (dB) between the high and low percentiles of the RMS of frames of a chunk, ignoring frames quieter than
    silence_db."""
    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, low: float = 0.1, high: float = 0.95, silence_db: float = -60.,
                 keep_channels: bool = False, sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, keep_channels=keep_channels, sr=sr, **kwargs)
        self.low, self.high, self.silence_db = low, high, silence_db

    @property
    def default_feature_name(self): return "dynamic_range"

    def compute(self, x):
        rms_db = 20 * torch.log10(torch.sqrt(_frames(x, self.frame_length, self.hop_length).square().mean(-1)) + _EPS)
        active = rms_db > self.silence_db
        masked = torch.where(active, rms_db, torch.full_like(rms_db, float("nan")))
        high, low = torch.nanquantile(masked, self.high, dim=-1), torch.nanquantile(masked, self.low, dim=-1)
        return torch.where(active.any(-1), high - low, torch.zeros_like(high))


@gin.configurable(module="features")
class SilenceRatio(_ChunkFeature):
    """ratio of frames whose RMS is below threshold_db."""
    def __init__(self, frame_length: int = 2048, hop_length: int | None = None, threshold_db: float = -60., keep_channels: bool = False,
                 sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, keep_channels=keep_channels, sr=sr, **kwargs)
        self.threshold_db = threshold_db

    @property
    def default_feature_name(self): return "silence_ratio"

    def compute(self, x):
        rms_db = 20 * torch.log10(torch.sqrt(_frames(x, self.frame_length, self.hop_length).square().mean(-1)) + _EPS)
        return (rms_db < self.threshold_db).float().mean(-1)


__all__ = ['ClippingRatio', 'DCOffset', 'DynamicRange', 'SilenceRatio']
