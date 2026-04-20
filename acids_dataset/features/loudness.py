import gin
import torch

from . import RAISE_EXC_IF_FEATURE_ERROR
from .base import AcidsDatasetFeature
from ..utils import loudness, frame_with_pad


@gin.configurable(module="features")
class Loudness(AcidsDatasetFeature):
    def __init__(
            self, 
            frame_length: int | None = None,
            keep_channels: bool = True,
            sr: int = 44100, 
            **kwargs
    ):
        super().__init__(**kwargs)
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

    #TODO transfer frame_length and keep_channels arguments to loudness
    def __call__(self, data) -> torch.Tensor:
        # if self.frame_length is None:
        #     if self.keep_channels:
        #         data_loudness = torch.zeros(*data.shape[:-1])
        #         for c in range(data.shape[-2]):
        #             data_loudness[..., c] = loudness(data[..., [c], :], self.sr)
        #     else:
        #         data_loudness = loudness(data, self.sr)
        # else:
        #     data_framed = frame_with_pad(data, frame_length=self.frame_length, hop_size=self.frame_length)
        #     if not self.keep_channels: 
        #         data_loudness = torch.zeros(*data_framed.shape[:-3], data_framed.shape[-1])
        #         for f in range(data_framed.shape[-1]):
        #             data_loudness[..., f] = loudness(data_framed[..., f], self.sr)
        #     else:
        #         data_loudness = torch.zeros(*data_framed.shape[:-2], data_framed.shape[-1])
        #         for f in range(data_framed.shape[-1]):
        #             for c in range(data_framed.shape[-3]):
        #                 data_loudness[..., c, f] = loudness(data_framed[..., c, :, f], self.sr)
        data_loudness = loudness(data, sample_rate=self.sr, frame_length=self.frame_length, keep_channels=self.keep_channels)
        return data_loudness
       