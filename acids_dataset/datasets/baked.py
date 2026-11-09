"""Baked features of online datasets : features computed once on whole audio files, and stored in .ad files (one per
audio file, mirroring the architecture of dataset folders). Chunks then read the frames of their position instead of
computing features on the fly."""
import os
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch

from ..features import AcidsDatasetFeature

BAKED_FOLDER = "ad-features"
BAKED_EXT = ".ad"
BAKED_VERSION = 1
_METADATA_KEY = "__metadata__"


def baked_file_path(roots: Dict[str, Path], audio_path: str) -> Path:
    """location of the baked file of an audio file, given as "<dataset folder name>/<path relative to the folder>"
    (as in fragments), for roots {dataset folder name: baked folder}."""
    dataset_name, _, rel_path = str(audio_path).partition("/")
    if dataset_name not in roots:
        raise KeyError("no baked folder for dataset %s (available : %s)"%(dataset_name, list(roots)))
    return Path(roots[dataset_name]) / (rel_path + BAKED_EXT)


def read_baked_file(path: Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """returns (metadata, {feature name: array}) of a baked file."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data[_METADATA_KEY]))
        arrays = {name: data[info['key']] for name, info in metadata['features'].items()}
    return metadata, arrays


def read_baked_metadata(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data[_METADATA_KEY]))


def write_baked_file(path: Path, metadata: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
    """writes a baked file (numpy .npz format, without pickles), atomically."""
    metadata = dict(metadata, features={name: dict(info, key="feature_%d"%i) for i, (name, info) in enumerate(metadata['features'].items())})
    os.makedirs(path.parent, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez(f, **{_METADATA_KEY: np.array(json.dumps(metadata))},
                 **{metadata['features'][name]['key']: np.asarray(array) for name, array in arrays.items()})
    os.replace(tmp_path, path)


def n_frames_for_length(n_samples: int, hop: float, frame_offset: int) -> int:
    """number of frames given by a feature on n_samples samples : floor(n_samples / hop) + frame_offset (at least 1)."""
    return max(1, int(math.floor(n_samples / hop + 1e-6)) + frame_offset)


def slice_frames(array: np.ndarray, start: float, length: float, sr: int, feature_sr: float, frame_offset: int) -> np.ndarray:
    """frames of a baked feature (time last) for a chunk of length seconds at start seconds : frames from the closest
    frame of start, as many as the feature gives on the chunk. Frames exceeding the file repeat the last frame."""
    hop = sr / feature_sr
    n = n_frames_for_length(int(round(length * sr)), hop, frame_offset)
    first = min(int(round(start * feature_sr)), max(array.shape[-1] - 1, 0))
    out = array[..., first:first + n]
    if out.shape[-1] < n:
        out = np.concatenate([out, np.repeat(out[..., -1:], n - out.shape[-1], axis=-1)], axis=-1)
    return out


def probe_frame_rate(compute, sr: int, n_channels: int, hop_hint: float | None = None) -> Tuple[float, int]:
    """frame rate (frames per second) and frame offset of a feature, such that it gives
    floor(n_samples / hop) + frame_offset frames on n_samples samples (hop = sr / frame rate). compute(n_samples)
    returns the feature array (time last) of a test signal of n_samples samples. Raises a ValueError if the feature
    has no time axis."""
    lengths = [2 ** 16, 2 ** 17]
    n1, n2 = (compute(l).shape[-1] for l in lengths)
    if n1 == n2:
        raise ValueError("its outputs have no time axis (same number of frames for %d and %d samples)"%tuple(lengths))
    hop = (lengths[1] - lengths[0]) / (n2 - n1)
    if hop_hint is not None and abs(hop_hint - hop) < 0.05 * hop:
        hop = float(hop_hint)
    elif abs(round(hop) - hop) < 0.05 * hop:
        hop = float(round(hop))
    # frame offset, on lengths multiple of the hop size
    k = 16
    na, nb = compute(int(round(k * hop))).shape[-1], compute(int(round(2 * k * hop))).shape[-1]
    if nb - na != k:
        raise ValueError("could not determine its frame rate (hop of %.2f samples, %d frames for %d hops)"%(hop, nb - na, k))
    return sr / hop, int(na - k)


def hop_hint(feature: AcidsDatasetFeature) -> float | None:
    """hop size of a feature, from its attributes if available."""
    for owner in [feature, getattr(feature, "mel_spectrogram", None)]:
        hop = getattr(owner, "hop_length", None)
        if isinstance(hop, (int, float)) and hop > 0:
            return float(hop)
    kwargs = getattr(feature, "kwargs", None)
    if isinstance(kwargs, dict) and isinstance(kwargs.get("hop_length"), (int, float)):
        return float(kwargs["hop_length"])
    return None


class BakedFeature(AcidsDatasetFeature):
    """feature read from baked files : the frames corresponding to the position of a fragment (start_pos and length of
    its metadata) are sliced from the feature computed on the whole file. Baked features are given by
    OnlineAudioDataset.use_baked_features."""
    supports_batch = False
    has_hash = False

    def __init__(self, name: str, roots: Dict[str, str | Path], cache_size: int = 4):
        self.roots = {k: str(v) for k, v in roots.items()}
        self.cache_size = cache_size
        self._cache = OrderedDict()
        super().__init__(name=name, device="cpu")

    def __repr__(self):
        return "BakedFeature(%s)"%self.feature_name

    def __getstate__(self):
        # cached arrays are not sent to workers
        state = dict(self.__dict__)
        state['_cache'] = OrderedDict()
        return state

    def to(self, device=None):
        return self

    def baked_path(self, audio_path: str) -> Path:
        return baked_file_path({k: Path(v) for k, v in self.roots.items()}, audio_path)

    def load(self, audio_path: str) -> Tuple[Dict[str, Any], np.ndarray]:
        """returns (metadata, array) of the feature for an audio file."""
        if audio_path in self._cache:
            self._cache.move_to_end(audio_path)
            return self._cache[audio_path]
        path = self.baked_path(audio_path)
        if not path.exists():
            raise FileNotFoundError("baked file %s not found : bake feature %s with dataset.bake_features"%(path, self.feature_name))
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data[_METADATA_KEY]))
            if self.feature_name not in metadata['features']:
                raise KeyError("feature %s is not baked in %s (baked features : %s)"%(self.feature_name, path, list(metadata['features'])))
            array = data[metadata['features'][self.feature_name]['key']]
        self._cache[audio_path] = (metadata, array)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return metadata, array

    def from_fragment(self, fragment, write: bool = True):
        fragment_metadata = fragment.get_metadata()
        metadata, array = self.load(fragment_metadata['audio_path'])
        info = metadata['features'][self.feature_name]
        out = slice_frames(array, float(fragment_metadata.get('start_pos', 0.)), float(fragment_metadata['length']),
                           metadata['sr'], info['feature_sr'], info['frame_offset'])
        if write:
            fragment.put_array(self.feature_name, out)
        return out

    def __call__(self, *args, **kwargs):
        raise RuntimeError("baked feature %s is read from fragments of an online dataset, and cannot be computed on audio"%self.feature_name)


def to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


__all__ = ['BakedFeature', 'BAKED_FOLDER', 'BAKED_EXT']
