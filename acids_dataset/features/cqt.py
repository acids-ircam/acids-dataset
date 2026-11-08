from typing import Optional, Callable
import numpy as np
import torch, gin.torch
import librosa

from .base import AcidsDatasetFeature

cqt_config = """
{{NAME}}:
    hop_length=512
    fmin=None 
    n_bins=84
    bins_per_octave=12
    tuning=0.0
    filter_scale=1
    norm=1
    sparsity=0.01
    window='hann'
    scale=True
    pad_mode='constant'
    res_type='soxr_hq'

features.parse_features:
    feature = @features.{{NAME}}
"""

_configs = {
    "cqt": cqt_config
}

@gin.configurable(module="features")
class CQT(AcidsDatasetFeature):
    def __init__(
            self, 
            name: str = None,
            hash_from_feature: Optional[Callable] = None, 
            device: torch.device = None,
            sr: int = 44100, 
            **kwargs
    ):
        self.cqt = librosa.cqt
        self.sr = sr
        self.kwargs = kwargs
        super().__init__(name=name, hash_from_feature=hash_from_feature, device=device)

    def __repr__(self):
        kwargs_repr = ", ".join([f"{a}={b}" for a, b in self.kwargs.items()])
        return f"CQT(sr={self.sr}, {kwargs_repr})"

    supports_batch = True

    @property
    def has_hash(self):
        return False

    @property
    def default_feature_name(self):
        return f"cqt_{self.kwargs.get('n_bins', 84)}"

    def _compute(self, data) -> np.ndarray:
        """constant-Q transform of (..., C, T) audio, with real and imaginary parts concatenated along the channel axis."""
        if torch.is_tensor(data): data = data.detach().cpu().numpy()
        cqt = self.cqt(np.asarray(data, dtype=np.float32), sr=self.sr, **self.kwargs)
        # (..., C, F, T) : channels are the third last axis
        return np.concatenate([cqt.real, cqt.imag], axis=-3).astype(np.float32)

    def from_fragment(self, fragment, write: bool = True):
        data = fragment.get_audio("waveform")
        try: 
            cqt = self._compute(data)
        except RuntimeError:
            return  
        if write:
            fragment.put_array(self.feature_name, cqt)
        return cqt

    def __call__(self, x):
        return torch.from_numpy(self._compute(x)).to(x.device if torch.is_tensor(x) else "cpu")
        
