import gin
import numpy as np
import torch
from typing import Optional, Callable
from torchaudio.transforms import MelSpectrogram
from . import RAISE_EXC_IF_FEATURE_ERROR
from .base import AcidsDatasetFeature


@gin.configurable(module="features")
class Mel(AcidsDatasetFeature):
    def __init__(
            self, 
            name: str = None,
            hash_from_feature: Optional[Callable] = None, 
            device: torch.device = None,
            sr: int = 44100, 
            **kwargs
    ):
        self.sr = sr
        self.mel_spectrogram = MelSpectrogram(**kwargs, sample_rate=sr)
        super().__init__(name=name, hash_from_feature=hash_from_feature, device=device)

    def __repr__(self):
        return f"Mel(n_mels = {self.mel_spectrogram.n_mels}, n_fft = {self.mel_spectrogram.n_fft}, sr={self.sr})"

    @property
    def has_hash(self):
        return False

    @property
    def default_feature_name(self):
        return f"mel_{self.mel_spectrogram.n_mels}"

    def from_fragment(self, fragment, write: bool = True):
        data = fragment.get_audio("waveform")
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        try: 
            mels = self.mel_spectrogram(data)
        except RuntimeError as e:
            if RAISE_EXC_IF_FEATURE_ERROR: raise e
            return  
        if write:
            fragment.put_array(self.feature_name, mels)
        return mels

    def __call__(self, x):
        return self.mel_spectrogram(x)
        




