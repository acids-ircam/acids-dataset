"""Quality and dynamics descriptors per chunk : clipping ratio, DC offset, dynamic range and silence ratio (see also
acids_dataset.health for reports on whole datasets)."""
import gin
import math
import torch
from typing import Sequence

from .framed import _EPS, _frames, _ChannelFeature
from ..utils import a_weighting, mean_square


class _ChunkFeature(_ChannelFeature):
    """descriptors of whole chunks : output (...,), or (..., C) with keep_channels."""
    _channel_axis = -1


@gin.configurable(module="ad.features")
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


@gin.configurable(module="ad.features")
class Peak(_ChunkFeature):
    """maximum absolute amplitude of a chunk, linear (db=False) or in dBFS (db=True) : of each channel with keep_channels,
    the maximum over channels otherwise (peaks are measured on channels, as mixes would hide them)."""
    def __init__(self, db: bool = False, keep_channels: bool = False, sr: int = 44100, **kwargs):
        super().__init__(frame_length=1, keep_channels=keep_channels, sr=sr, **kwargs)
        self.db = db

    @property
    def default_feature_name(self): return "peak"

    def _channels(self, data):
        from .framed import _as_tensor
        return _as_tensor(data)

    def compute(self, x):
        peak = x.abs().amax(-1)
        if not self.keep_channels:
            peak = peak.amax(-1, keepdim=True)
        return 20 * torch.log10(peak + _EPS) if self.db else peak


@gin.configurable(module="ad.features")
class DCOffset(_ChunkFeature):
    """mean value of the signal (of each channel with keep_channels, of the mix otherwise)."""
    def __init__(self, keep_channels: bool = True, sr: int = 44100, **kwargs):
        super().__init__(frame_length=1, keep_channels=keep_channels, sr=sr, **kwargs)

    @property
    def default_feature_name(self): return "dc_offset"

    def compute(self, x):
        return x.mean(-1)


@gin.configurable(module="ad.features")
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


@gin.configurable(module="ad.features")
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


@gin.configurable(module="ad.features")
class LoudnessMeter(_ChunkFeature):
    """loudness statistics of a chunk, computed on the levels of its frames : RMS level in dBFS (db), A-weighted RMS
    level in dB(A) (dba, IEC 61672 weighting, 0 dB at 1 kHz), and loudness in LUFS (lufs, ITU-R BS.1770 K-weighting,
    see utils.loudness). Requested properties are concatenated in the last dimension, in the order they are given 
    (see output_names) :
        - min_db, max_db, min_dba, max_dba, min_lufs, max_lufs : minimum / maximum level of frames (1 value each).
        - db_quantiles, dba_quantiles, lufs_quantiles : quantiles of frame levels at n_quantiles evenly spaced 
            probabilities from 0 to 1 (n_quantiles values each : the first one is the minimum, the last one the maximum).
    Levels of silent frames are floored to floor_db (in LUFS, frames below the absolute gate of -70 LUFS are silent).
    Output is (..., n_values), or (..., C, n_values) with keep_channels (levels of each channel). Without keep_channels,
    db and dba levels are the ones of the mix of channels, and lufs the combined loudness of channels (BS.1770, up to 5
    channels).

    Args:
        properties (Sequence[str] | str): requested properties. Defaults to ("min_db", "max_db", "min_dba", "max_dba").
        n_quantiles (int): number of quantiles of db_quantiles and dba_quantiles. Defaults to 10.
        frame_length (int): frame length (samples). Defaults to 2048.
        hop_length (int | None): hop between frames (samples) ; frame_length if None. Defaults to None.
        floor_db (float): minimum level (dB). Defaults to -120.
    """
    available_properties = ["min_db", "max_db", "min_dba", "max_dba", "min_lufs", "max_lufs", "db_quantiles", "dba_quantiles", "lufs_quantiles"]
    _channel_axis = -2

    def __init__(self, properties: Sequence[str] | str = ("min_db", "max_db", "min_dba", "max_dba"), n_quantiles: int = 10,
                 frame_length: int = 2048, hop_length: int | None = None, floor_db: float = -120., keep_channels: bool = False,
                 sr: int = 44100, **kwargs):
        super().__init__(frame_length=frame_length, hop_length=hop_length, keep_channels=keep_channels, sr=sr, **kwargs)
        properties = [properties] if isinstance(properties, str) else list(properties)
        unknown = [p for p in properties if p not in self.available_properties]
        if len(properties) == 0 or unknown:
            raise ValueError("properties must be in %s, got %s"%(self.available_properties, properties))
        if n_quantiles < 1:
            raise ValueError("n_quantiles must be positive, got %s"%n_quantiles)
        self.properties = properties
        self.n_quantiles = int(n_quantiles)
        self.floor_db = float(floor_db)

    @property
    def default_feature_name(self): return "loudness_meter"

    def __repr__(self):
        return "LoudnessMeter(properties=%s, n_quantiles=%d, frame_length=%d, hop_length=%d, floor_db=%s, sr=%s)"%(
            self.properties, self.n_quantiles, self.frame_length, self.hop_length, self.floor_db, self.sr)

    @property
    def output_names(self):
        """names of the values of the last output dimension."""
        names = []
        for p in self.properties:
            if p.endswith("_quantiles"):
                probs = torch.linspace(0, 1, self.n_quantiles) if self.n_quantiles > 1 else torch.tensor([0.5])
                names.extend("%s_q%g"%(p.split("_")[0], round(float(q), 4)) for q in probs)
            else:
                names.append(p)
        return names

    def _channels(self, data):
        # channels are mixed in compute (LUFS combines the loudness of channels)
        from .framed import _as_tensor
        return _as_tensor(data)

    def _lufs(self, frames: torch.Tensor) -> torch.Tensor:
        """loudness of frames (..., C, n_frames, frame_length) -> (..., C', n_frames) (C' = 1 without keep_channels)."""
        from ..utils import loudness
        lufs = loudness(frames.movedim(-3, -2), self.sr, keep_channels=self.keep_channels)
        lufs = lufs.movedim(-1, -2) if self.keep_channels else lufs.unsqueeze(-2)
        return torch.nan_to_num(lufs, neginf=self.floor_db).clamp(min=self.floor_db)

    def _levels(self, frames: torch.Tensor, weighted: bool) -> torch.Tensor:
        """dB levels of frames (..., n_frames, frame_length) -> (..., n_frames)."""
        if not self.keep_channels:
            frames = frames.mean(-3, keepdim=True)
        power = mean_square(frames, self.sr, a_weighted=weighted)
        return 10 * torch.log10(power.clamp(min=10 ** (self.floor_db / 10)))

    def compute(self, x):
        frames = _frames(x, self.frame_length, self.hop_length)
        levels = {}
        outs = []
        for p in self.properties:
            kind = p.replace("min_", "").replace("max_", "").replace("_quantiles", "")
            if kind not in levels:
                levels[kind] = self._lufs(frames) if kind == "lufs" else self._levels(frames, weighted=kind == "dba")
            l = levels[kind]
            if p.startswith("min"):
                outs.append(l.amin(-1, keepdim=True))
            elif p.startswith("max"):
                outs.append(l.amax(-1, keepdim=True))
            else:
                probs = torch.linspace(0, 1, self.n_quantiles) if self.n_quantiles > 1 else torch.tensor([0.5])
                q = torch.quantile(l, probs.to(l.device, l.dtype), dim=-1)
                outs.append(q.movedim(0, -1))
        return torch.cat(outs, -1)


__all__ = ['LoudnessMeter', 'Peak', 'ClippingRatio', 'DCOffset', 'DynamicRange', 'SilenceRatio']
