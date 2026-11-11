"""Ambisonics : encoding of multichannel signals into spherical harmonics, and decoding of ambisonic signals to
loudspeaker layouts. Ambisonic signals follow the AmbiX convention : ACN channel ordering, real spherical harmonics
without Condon-Shortley phase, and SN3D (default) or N3D normalization.

Directions are given in degrees : azimuth counterclockwise from the front (90 : left), elevation upwards from the
horizontal plane (90 : top)."""
import math
import gin
import numpy as np
import torch
from scipy.special import lpmv

from .base import Transform

NORMALIZATIONS = ["sn3d", "n3d"]


def _as_tensor(data) -> torch.Tensor:
    if not torch.is_tensor(data):
        data = torch.from_numpy(np.asarray(data))
    data = data.float() if not data.is_floating_point() else data
    return data[None] if data.ndim == 1 else data

# loudspeaker layouts, as (azimuth, elevation) in degrees
LAYOUTS = {
    "mono": [(0, 0)],
    "stereo": [(30, 0), (-30, 0)],
    "quad": [(45, 0), (135, 0), (-135, 0), (-45, 0)],
    "5.0": [(30, 0), (-30, 0), (0, 0), (110, 0), (-110, 0)],
    "7.0": [(30, 0), (-30, 0), (0, 0), (90, 0), (-90, 0), (150, 0), (-150, 0)],
    "octagon": [(az, 0) for az in range(0, 360, 45)],
    "tetrahedron": [(45, 35.264), (-135, 35.264), (135, -35.264), (-45, -35.264)],
    "cube": [(az, el) for el in (35.264, -35.264) for az in (45, 135, -135, -45)],
}


def ambisonic_channels(order: int) -> int:
    """number of ambisonic channels of an order : (order + 1) ** 2."""
    return (order + 1) ** 2


def ambisonic_order(n_channels: int) -> int:
    """ambisonic order of a signal with (order + 1) ** 2 channels ; raises a ValueError otherwise."""
    order = int(round(math.sqrt(n_channels))) - 1
    if ambisonic_channels(order) != n_channels:
        raise ValueError("ambisonic signals have (order + 1)^2 channels (1, 4, 9, 16, ...), got %d"%n_channels)
    return order


def spherical_harmonics(order: int, azimuth, elevation, normalization: str = "sn3d") -> np.ndarray:
    """real spherical harmonics up to order, in ACN order, without Condon-Shortley phase (AmbiX convention), for
    directions given in degrees. Returns an array (n_directions, (order + 1)^2). With SN3D normalization, the first
    order channels of a source at (azimuth, elevation) are W = 1, Y = sin(az) cos(el), Z = sin(el), X = cos(az) cos(el)."""
    assert normalization in NORMALIZATIONS, "normalization must be in %s, got %s"%(NORMALIZATIONS, normalization)
    azimuth = np.deg2rad(np.atleast_1d(np.asarray(azimuth, dtype=np.float64)))
    elevation = np.deg2rad(np.atleast_1d(np.asarray(elevation, dtype=np.float64)))
    sin_el = np.sin(elevation)
    out = np.zeros((len(azimuth), ambisonic_channels(order)))
    for l in range(order + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            # scipy's lpmv includes the Condon-Shortley phase (-1)^m, removed here
            legendre = (-1) ** am * lpmv(am, l, sin_el)
            norm = math.sqrt((2 - (m == 0)) * math.factorial(l - am) / math.factorial(l + am))
            if normalization == "n3d":
                norm *= math.sqrt(2 * l + 1)
            angular = np.cos(am * azimuth) if m >= 0 else np.sin(am * azimuth)
            out[:, l * l + l + m] = norm * legendre * angular
    return out


def fibonacci_sphere(n: int) -> np.ndarray:
    """n directions (azimuth, elevation) in degrees, spread regularly on the sphere (Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    elevation = np.rad2deg(np.arcsin(1 - 2 * i / n))
    azimuth = np.rad2deg((math.pi * (1 + 5 ** 0.5) * i) % (2 * math.pi))
    azimuth = (azimuth + 180) % 360 - 180
    return np.stack([azimuth, elevation], axis=-1)


def get_layout(layout, n_channels: int | None = None) -> np.ndarray:
    """directions (n, 2) in degrees of a layout : a name of LAYOUTS, a list of (azimuth, elevation), or None. With None,
    the layout is chosen from n_channels : mono, stereo, tetrahedron (4 channels, first order ambisonics), or n_channels
    directions equally spaced on the horizontal plane
    (counterclockwise from the front)."""
    if layout is None:
        if n_channels is None:
            raise ValueError("layout=None needs a number of channels")
        defaults = {1: "mono", 2: "stereo", 4: "tetrahedron"}
        if n_channels in defaults:
            return np.asarray(LAYOUTS[defaults[n_channels]], dtype=np.float64)
        azimuth = (np.arange(n_channels) * 360. / n_channels + 180) % 360 - 180
        return np.stack([azimuth, np.zeros(n_channels)], axis=-1)
    if isinstance(layout, str):
        if layout not in LAYOUTS:
            raise ValueError("unknown layout %s ; available layouts : %s"%(layout, list(LAYOUTS)))
        layout = LAYOUTS[layout]
    directions = np.asarray(layout, dtype=np.float64).reshape(-1, 2)
    return directions


def _check_order(order):
    if not (order == "adaptive" or (isinstance(order, int) and order >= 0)):
        raise ValueError('order must be "adaptive" or a positive integer, got %s'%(order,))


def _rank(matrix: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(matrix, tol=1e-8 * math.sqrt(matrix.size)))


def _fits(x: torch.Tensor, matrix: np.ndarray, tolerance: float = 1e-3) -> bool:
    """True if signals x (..., M, T) are in the span of the columns of matrix (M, k) (relative residual of the
    projection below tolerance)."""
    m = torch.as_tensor(matrix, dtype=torch.float64, device=x.device)
    x = x.to(torch.float64)
    residual = x - m @ (torch.linalg.pinv(m) @ x)
    return float(residual.norm()) <= tolerance * max(float(x.norm()), 1e-12)


def _select(x: torch.Tensor, candidates, matrix_fn, what: str):
    """the first candidate whose matrix spans signals x (candidates are sorted by increasing size)."""
    if len(candidates) == 1:
        return candidates[0]
    for candidate in candidates:
        if _fits(x, matrix_fn(candidate)):
            return candidate
    raise ValueError("%s cannot be inferred from signals (candidates : %s)"%(what, candidates))


@gin.configurable(module="ad.transforms")
class AmbisonicEncoder(Transform):
    """Encodes multichannel signals into ambisonics : (..., C, T) -> (..., (N + 1)^2, T). Each input channel is a
    source in the direction given by layout (a name of LAYOUTS, a list of (azimuth, elevation) in degrees, or None : mono,
    stereo, tetrahedron for 4 channels, or C directions equally spaced on the horizontal plane), and ambisonic signals are the sum of sources
    weighted by their spherical harmonics (ACN order, SN3D or N3D normalization).

    order is "adaptive" (default) : the order is inferred from the number of channels, as the lowest order whose
    encoding is invertible for the layout (the C sources give C independent ambisonic components). For instance, mono
    gives order 0, stereo and tetrahedra order 1, 5.0 and quad order 2, 7.0 order 3 (horizontal layouts of C channels
    need order ceil((C - 1) / 2)). With adaptive orders and default layouts, signals with different numbers of channels
    can be encoded by the same transform. With a fixed order N (an integer), inputs whose channels cannot be encoded
    invertibly at order N (e.g. 7.0 signals at order 1) raise a ValueError.

    With strict=False, fixed orders also encode channels that cannot be encoded invertibly (e.g. loudspeaker signals,
    as in AmbisonicDecoder.invert) : sources are then mixed in the ambisonic signals.

    invert decodes ambisonic signals back to the input channels with an AmbisonicDecoder (mode matching) in the
    directions of the input channels (see inverse) : inversion is exact, as encodings are invertible. The order is
    inferred from the number of ambisonic channels ((N + 1)^2, or a ValueError is raised ; with a fixed order, other
    numbers of channels raise a ValueError), and the number of channels from the layout. With layout=None, several
    numbers of channels can give the same order (e.g. stereo and 3 channels give order 1) : the number of channels is
    the lowest one whose encoding gives the signals (or n_channels, if given).
    """
    invertible = True
    allow_random = False
    takes_as_input = Transform.input_types.torch

    def __init__(self, order: int | str = "adaptive", layout: str | list | None = None, normalization: str = "sn3d",
                 strict: bool = True, sr: int | None = None, name: str | None = None, **kwargs):
        _check_order(order)
        assert normalization in NORMALIZATIONS, "normalization must be in %s, got %s"%(NORMALIZATIONS, normalization)
        super().__init__(sr=sr, name=name or "ambisonics", **kwargs)
        self.order, self.layout, self.normalization, self.strict = order, layout, normalization, strict

    def __repr__(self):
        return "AmbisonicEncoder(order=%s, layout=%s, normalization=%s)"%(self.order, self.layout, self.normalization)

    def _directions(self, n_channels: int) -> np.ndarray:
        directions = get_layout(self.layout, n_channels)
        if len(directions) != n_channels:
            raise ValueError("layout has %d directions, but inputs have %d channels"%(len(directions), n_channels))
        return directions

    def _harmonics(self, order: int, n_channels: int) -> np.ndarray:
        directions = self._directions(n_channels)
        return spherical_harmonics(order, directions[:, 0], directions[:, 1], self.normalization)

    def get_order(self, n_channels: int) -> int:
        """ambisonic order used for inputs with n_channels channels : inferred with adaptive orders, and checked
        with fixed orders (raises a ValueError if the channels cannot be encoded invertibly)."""
        if self.order != "adaptive":
            if getattr(self, "strict", True) and _rank(self._harmonics(self.order, n_channels)) != n_channels:
                raise ValueError("%d channels (layout %s) cannot be encoded invertibly at order %d"%(n_channels, self.layout, self.order))
            return self.order
        # C distinct directions are always separated by harmonics of order C - 1
        for order in range(int(math.ceil(math.sqrt(n_channels))) - 1, n_channels):
            if _rank(self._harmonics(order, n_channels)) == n_channels:
                return order
        raise ValueError("layout %s has coincident directions : its encoding cannot be inverted"%(self.layout,))

    def encoding_matrix(self, n_channels: int) -> np.ndarray:
        """encoding matrix ((N + 1)^2, C) for inputs with n_channels channels."""
        return self._harmonics(self.get_order(n_channels), n_channels).T

    def apply(self, data) -> torch.Tensor:
        x = _as_tensor(data)
        matrix = torch.as_tensor(self.encoding_matrix(x.shape[-2]), dtype=x.dtype, device=x.device)
        return matrix @ x

    def _order_or_none(self, n_channels: int) -> int | None:
        try:
            return self.get_order(n_channels)
        except ValueError:
            return None

    def inverse(self, n_channels: int) -> "AmbisonicDecoder":
        """decoder inverting the encoding of n_channels channels : AmbisonicDecoder (mode matching) at the order of the
        encoding, whose loudspeakers are the directions of the input channels."""
        return AmbisonicDecoder(layout=self._directions(n_channels).tolist(), order=self.get_order(n_channels),
                                normalization=self.normalization, method="mode_matching", sr=self.sr)

    def invert(self, data, n_channels: int | None = None) -> torch.Tensor:
        """input channels (..., C, T) from ambisonic signals (..., (N + 1)^2, T), decoded with inverse(C). The number of
        channels C is given by the layout, by n_channels if given, or inferred from signals (layout=None). With
        layout=None, signals that are not encodings of default layouts (e.g. second-order recordings with vertical
        components) are decoded to (N + 1)^2 sources spread regularly on the sphere (as AmbisonicDecoder(layout=None)) :
        encoding them with these directions gives the signals back."""
        a = _as_tensor(data)
        order = ambisonic_order(a.shape[-2])
        if self.order != "adaptive" and order != self.order:
            raise ValueError("order %d needs %d ambisonic channels, got %d"%(self.order, ambisonic_channels(self.order), a.shape[-2]))
        if n_channels is None:
            if self.layout is not None:
                n_channels = len(get_layout(self.layout))
            else:
                candidates = [c for c in range(1, a.shape[-2] + 1) if self._order_or_none(c) == order]
                # the lowest number of channels whose encoding gives the signals
                fitting = [c for c in candidates if _fits(a, self.encoding_matrix(c))]
                if len(fitting) > 0:
                    n_channels = fitting[0]
                else:
                    # signals are not encodings of default layouts (e.g. full-sphere recordings, with vertical
                    # components) : decoded to (N + 1)^2 virtual sources spread regularly on the sphere (exact)
                    return AmbisonicDecoder(layout=None, order=order, normalization=self.normalization,
                                            method="mode_matching", sr=self.sr).apply(a)
        if ambisonic_channels(self.get_order(n_channels)) != a.shape[-2]:
            raise ValueError("encoding of %d channels gives %d ambisonic channels, got %d"%(
                n_channels, ambisonic_channels(self.get_order(n_channels)), a.shape[-2]))
        return self.inverse(n_channels).apply(a)


@gin.configurable(module="ad.transforms")
class AmbisonicDecoder(Transform):
    """Decodes ambisonic signals (..., K, T), in ACN order and SN3D or N3D normalization, to the loudspeakers of layout :
    (..., L, T).

    order is "adaptive" (default) : the order is inferred from the number of input channels (K = (N + 1)^2, otherwise a
    ValueError is raised). With a fixed order N (an integer), inputs that do not have (N + 1)^2 channels raise a
    ValueError.

    layout is a name of LAYOUTS, a list of (azimuth, elevation) in degrees, or None : (N + 1)^2 virtual loudspeakers
    spread regularly on the sphere (Fibonacci lattice).

    Methods :
        - "mode_matching" (default) : pseudo-inverse of the encoding matrix of the layout ; decoding signals encoded
          with AmbisonicEncoder on the same layout gives the original signals back (if the order is high enough).
        - "sampling" : spherical harmonics of each loudspeaker direction, scaled by 1 / L (best for regular layouts).

    invert encodes loudspeaker signals back to ambisonics. With mode matching, they are encoded with an
    AmbisonicEncoder whose sources are the loudspeakers (see inverse) ; sampling decoders are not inverted by encoders
    (except on regular layouts), and are inverted with the pseudo-inverse of their decoding matrix. The order is the
    fixed order, or is inferred from the number of loudspeakers : (N + 1)^2 loudspeakers with layout=None ; with a
    layout, the lowest order (with (N + 1)^2 <= L) whose decoding gives the signals (or order, if given). Inversion is
    exact if the layout determines every ambisonic component (at least (N + 1)^2 loudspeakers in independent
    directions, e.g. layout=None) ; horizontal layouts lose vertical components.
    """
    methods = ["mode_matching", "sampling"]
    invertible = True
    allow_random = False
    takes_as_input = Transform.input_types.torch

    def __init__(self, layout: str | list | None = None, order: int | str = "adaptive", normalization: str = "sn3d",
                 method: str = "mode_matching", sr: int | None = None, name: str | None = None, **kwargs):
        _check_order(order)
        assert normalization in NORMALIZATIONS, "normalization must be in %s, got %s"%(NORMALIZATIONS, normalization)
        assert method in self.methods, "method must be in %s, got %s"%(self.methods, method)
        super().__init__(sr=sr, name=name or "ambisonics_decoded", **kwargs)
        self.layout, self.order, self.normalization, self.method = layout, order, normalization, method

    def __repr__(self):
        return "AmbisonicDecoder(layout=%s, order=%s, normalization=%s, method=%s)"%(self.layout, self.order, self.normalization, self.method)

    def get_order(self, n_channels: int) -> int:
        """ambisonic order of inputs with n_channels channels : inferred with adaptive orders, and checked with fixed
        orders (raises a ValueError if n_channels is not (order + 1)^2)."""
        if self.order == "adaptive":
            return ambisonic_order(n_channels)
        if n_channels != ambisonic_channels(self.order):
            raise ValueError("order %d needs %d channels, got %d"%(self.order, ambisonic_channels(self.order), n_channels))
        return self.order

    def n_loudspeakers(self, order: int) -> int:
        return ambisonic_channels(order) if self.layout is None else len(get_layout(self.layout))

    def directions(self, order: int) -> np.ndarray:
        """directions (L, 2) of loudspeakers, in degrees."""
        return fibonacci_sphere(ambisonic_channels(order)) if self.layout is None else get_layout(self.layout)

    def inverse(self, order: int) -> "AmbisonicEncoder":
        """encoder inverting mode matching decodings at order : AmbisonicEncoder at this order, whose sources are the
        loudspeakers (strict=False, as layouts may have more loudspeakers than ambisonic channels). Sampling decoders
        are only inverted by this encoder on regular layouts."""
        return AmbisonicEncoder(order=order, layout=self.directions(order).tolist(), normalization=self.normalization,
                                strict=False, sr=self.sr)

    def decoding_matrix(self, order: int) -> np.ndarray:
        """decoding matrix (L, (order + 1)^2)."""
        directions = self.directions(order)
        harmonics = spherical_harmonics(order, directions[:, 0], directions[:, 1], self.normalization)    # (L, K)
        if self.method == "mode_matching":
            return np.linalg.pinv(harmonics.T)
        # sampling : projection on N3D harmonics, that are orthonormal on the sphere
        degrees = np.floor(np.sqrt(np.arange(ambisonic_channels(order))))
        to_n3d = np.sqrt(2 * degrees + 1) if self.normalization == "sn3d" else np.ones_like(degrees)
        n3d = harmonics * to_n3d
        return n3d * to_n3d / len(directions)

    def apply(self, data) -> torch.Tensor:
        x = _as_tensor(data)
        order = self.get_order(x.shape[-2])
        matrix = torch.as_tensor(self.decoding_matrix(order), dtype=x.dtype, device=x.device)
        return matrix @ x

    def invert(self, data, order: int | None = None) -> torch.Tensor:
        """ambisonic signals (..., (N + 1)^2, T) from loudspeaker signals (..., L, T). The order is the fixed order of
        the transform, order if given, or is inferred from the number of loudspeakers (and from signals, for layouts that
        can hold several orders)."""
        s = _as_tensor(data)
        n_speakers = s.shape[-2]
        if order is None:
            if self.order != "adaptive":
                order = self.order
            elif self.layout is None:
                order = ambisonic_order(n_speakers)
            else:
                if n_speakers != len(get_layout(self.layout)):
                    raise ValueError("layout has %d loudspeakers, got %d channels"%(len(get_layout(self.layout)), n_speakers))
                candidates = [n for n in range(int(math.sqrt(n_speakers))) if ambisonic_channels(n) <= n_speakers]
                order = _select(s, candidates, self.decoding_matrix, "order")
        if len(self.directions(order)) != n_speakers:
            raise ValueError("decoding at order %d gives %d loudspeakers, got %d channels"%(order, len(self.directions(order)), n_speakers))
        if self.method == "mode_matching":
            # the encoding matrix of loudspeaker directions is the pseudo-inverse of the decoding matrix
            return self.inverse(order).apply(s)
        return torch.as_tensor(np.linalg.pinv(self.decoding_matrix(order)), dtype=s.dtype, device=s.device) @ s


__all__ = ['AmbisonicEncoder', 'AmbisonicDecoder', 'spherical_harmonics', 'get_layout', 'LAYOUTS']
