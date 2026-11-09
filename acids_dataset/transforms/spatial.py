"""Spatial transforms of multichannel signals : mid / side transform (see also ambisonics)."""
import gin
import numpy as np
import torch

from .base import Transform


@gin.configurable(module="transforms")
class MidSide(Transform):
    """Mid / side transform of stereo signals : (..., 2, T) -> (..., 2, T), mid first. Raises a ValueError if inputs do
    not have 2 channels.

    Normalizations :
        - "average" (default) : M = (L + R) / 2, S = (L - R) / 2 ; the inverse is L = M + S, R = M - S.
        - "orthonormal" : M = (L + R) / sqrt(2), S = (L - R) / sqrt(2). Energy preserving ; the transform is its own
          inverse.
    With inverse=True, inputs are mid / side signals, decoded into left / right signals (default name : leftright).
    invert gives inputs back exactly (decoding for mid / side signals, encoding with inverse=True). For a stereo width
    descriptor, see features.Width.
    """
    normalizations = ["average", "orthonormal"]
    invertible = True
    allow_random = False
    takes_as_input = Transform.input_types.torch

    def __init__(self, normalization: str = "average", inverse: bool = False, sr: int | None = None,
                 name: str | None = None, **kwargs):
        assert normalization in self.normalizations, "normalization must be in %s, got %s"%(self.normalizations, normalization)
        super().__init__(sr=sr, name=name or ("leftright" if inverse else "midside"), **kwargs)
        self.normalization = normalization
        self.inverse = inverse

    def __repr__(self):
        return "MidSide(normalization=%s, inverse=%s)"%(self.normalization, self.inverse)

    def apply(self, data) -> torch.Tensor:
        x = data if torch.is_tensor(data) else torch.from_numpy(np.asarray(data)).float()
        if x.ndim < 2 or x.shape[-2] != 2:
            raise ValueError("MidSide needs stereo inputs (..., 2, T), got %d channel(s)"%(x.shape[-2] if x.ndim > 1 else 1))
        if self.normalization == "orthonormal":
            scale = 2 ** -0.5
        else:
            scale = 1. if self.inverse else 0.5
        a, b = x[..., 0, :], x[..., 1, :]
        return torch.stack([a + b, a - b], dim=-2) * scale

    def invert(self, data) -> torch.Tensor:
        """left / right signals from mid / side signals (..., 2, T) (mid / side signals with inverse=True)."""
        return MidSide(normalization=self.normalization, inverse=not self.inverse, sr=self.sr).apply(data)


__all__ = ['MidSide']
