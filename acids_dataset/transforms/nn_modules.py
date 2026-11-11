"""TorchScript-compatible torch.nn.Module versions of transforms (see Transform.nn_module). Modules apply transforms
deterministically (the probability p of random transforms is ignored) ; modules of invertible transforms also have an
invert method (exported to TorchScript)."""
import math
from typing import List

import torch
import torchaudio


class TransformModule(torch.nn.Module):
    """base of transform modules : forward applies the transform. Modules of invertible transforms implement invert
    (decorated with torch.jit.export)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class IdentityModule(TransformModule):
    """identity : forward and invert give inputs back."""

    @torch.jit.export
    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ChannelMatrixModule(TransformModule):
    """linear transform of channels : (..., C, T) -> matrix @ x (..., C', T), and invert : inverse_matrix @ y. Matrices
    are stored in float64 and cast to the type of inputs."""

    def __init__(self, matrix: torch.Tensor, inverse_matrix: torch.Tensor):
        super().__init__()
        self.register_buffer("matrix", torch.as_tensor(matrix, dtype=torch.float64))
        self.register_buffer("inverse_matrix", torch.as_tensor(inverse_matrix, dtype=torch.float64))

    @staticmethod
    def _channels(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0) if x.dim() == 1 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._channels(x)
        return self.matrix.to(x) @ x

    @torch.jit.export
    def invert(self, x: torch.Tensor) -> torch.Tensor:
        x = self._channels(x)
        return self.inverse_matrix.to(x) @ x


class MidSideModule(TransformModule):
    """mid / side encoding of stereo signals (..., 2, T) (see transforms.MidSide)."""
    scale: float
    inverse_scale: float

    def __init__(self, scale: float, inverse_scale: float):
        super().__init__()
        self.scale = scale
        self.inverse_scale = inverse_scale

    @staticmethod
    def _apply(x: torch.Tensor, scale: float) -> torch.Tensor:
        if x.dim() < 2 or x.shape[-2] != 2:
            raise ValueError("MidSide needs stereo inputs (..., 2, T)")
        a, b = x[..., 0, :], x[..., 1, :]
        return torch.stack([a + b, a - b], dim=-2) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply(x, self.scale)

    @torch.jit.export
    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply(x, self.inverse_scale)


class PreEmphasisModule(TransformModule):
    coeff: float

    def __init__(self, coeff: float):
        super().__init__()
        self.coeff = coeff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torchaudio.functional.preemphasis(x, self.coeff)


class Log1pModule(TransformModule):
    contrast: float

    def __init__(self, contrast: float):
        super().__init__()
        self.contrast = contrast

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(1 + torch.clamp(x * self.contrast, 1e-5))


class Exp1mModule(TransformModule):
    contrast: float

    def __init__(self, contrast: float):
        super().__init__()
        self.contrast = contrast

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x) / self.contrast - 1


class NormalizeModule(TransformModule):
    """scales inputs to a peak of 0 dBFS, with a maximum gain of max_gain_db (see transforms.utils.normalize_signal)."""
    max_gain_db: float

    def __init__(self, max_gain_db: float = 30.):
        super().__init__()
        self.max_gain_db = max_gain_db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        peak = x.abs().amax()
        if bool(peak == 0):
            return x
        log_gain = torch.clamp(-20 * torch.log10(peak), max=self.max_gain_db)
        return x * 10 ** (log_gain / 20)


class LFilterModule(TransformModule):
    """IIR filter (torchaudio.functional.lfilter) of coefficients b (numerator) and a (denominator)."""

    def __init__(self, b: torch.Tensor, a: torch.Tensor):
        super().__init__()
        self.register_buffer("b", b)
        self.register_buffer("a", a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torchaudio.functional.lfilter(x, self.a.to(x), self.b.to(x), clamp=False)


class NormRangeModule(TransformModule):
    max_value: float
    boost: bool

    def __init__(self, max_value: float, boost: bool):
        super().__init__()
        self.max_value = max_value
        self.boost = boost

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape([1, 1, x.shape[-1]]) if x.dim() == 1 else x.reshape([-1, x.shape[-2], x.shape[-1]])
        peak = x_flat.abs().amax(dim=(-2, -1), keepdim=True)
        scale = torch.where(peak > 1e-10, self.max_value / peak.clamp(min=1e-10), torch.ones_like(peak))
        if not self.boost:
            scale = scale.clamp(max=1.)
        return (x_flat * scale).reshape(shape)


class ComposeModule(TransformModule):
    """sequentially applies modules."""

    def __init__(self, modules: List[torch.nn.Module]):
        super().__init__()
        self.transforms = torch.nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x


__all__ = ['TransformModule', 'IdentityModule', 'ChannelMatrixModule', 'MidSideModule', 'PreEmphasisModule', 'Log1pModule',
           'Exp1mModule', 'NormalizeModule', 'LFilterModule', 'NormRangeModule', 'ComposeModule']
