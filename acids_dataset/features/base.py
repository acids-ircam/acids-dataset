from typing import Optional, Callable, Any
from types import MethodType
import inspect
import os
import torch
import torchaudio
from collections import UserDict
from absl import logging
from ..utils import load_file, resolve_device, get_subclasses_from_package, generate_config_from_obj, parse_config_pattern

class FeatureException(Exception):
    pass

#TODO accelerate hashing with ordering
class FileHash(UserDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._idx = 0

    def __contains__(self, key):
        for i in self.data.keys():
            if i.endswith(key): return True
        return False

    @property
    def current_id(self):
        return int(self._idx)

    def __getitem__(self, key):
        for i in self.data.keys():
            if i.endswith(key): return self.data[i]
        return False

    def __setitem__(self, key: Any, item: Any) -> None:
        self._idx += 1
        return super().__setitem__(key, item)



default_feature_pattern = """
{{NAME}}:
\tdevice=%DEVICE
\t{{ARGS}}

ad.features.parse_features:
    features = @ad.features.{{NAME}}
"""

class AcidsDatasetFeature(object):
    # denylist = ['audio_path', 'audio_data', 'sr', 'start', 'end', 'duration']
    denylist = []
    has_hash = False
    dont_export_to_gin_config = ["self", "name", "args", "kwargs", "hash_from_feature", "device", "metadata"]
    def __init__(
            self, 
            name: Optional[str] = None,
            hash_from_feature: Optional[Callable] = None, 
            device: torch.device | str | None = None, 
            metadata = {}
        ):
        self.feature_name = name or self.default_feature_name
        # init hash function
        if hash_from_feature:
            self.hash_from_feature = MethodType(hash_from_feature, self)
        elif not hasattr(self, "hash_from_feature"):
            self.hash_from_feature = None
        if self.hash_from_feature is not None and not hasattr(self, "has_hash"):
            self.has_hash = True
        # metadata (for retaining clustering, or whatever)
        self.metadata = metadata
        # device handling
        self.device = device
        self.to(device)

    @classmethod
    def init_signature(cls): 
        return dict(inspect.signature(cls.__init__).parameters)

    def to(self, device = None):
        """sets the device of the feature (subclasses with modules also move them), and returns the feature. device is 
        a device or a device name, None for CPU, or "auto" for the default accelerated device (CUDA if available ; MPS 
        with USE_MPS=1, see utils.get_default_accelerated_device)."""
        self.device = resolve_device(device)
        return self

    @property
    def default_feature_name(self):
        return type(self).__name__.lower()

    def close(self):
        """if some features has side effects like buffering, empty buffers and delete files"""
        pass

    # invertible features give audio back from their outputs (see invert)
    invertible = False

    def invert(self, features, **kwargs):
        """audio from outputs of the feature, for invertible features (see invertible) ; raises a NotImplementedError
        otherwise."""
        raise NotImplementedError("%s is not invertible"%type(self).__name__)

    def nn_module(self, **kwargs) -> torch.nn.Module:
        """TorchScript-compatible torch.nn.Module computing the feature from audio (forward), with an invert method for
        invertible features. Raises a NotImplementedError for features without module (e.g. numpy / librosa based
        features)."""
        raise NotImplementedError("%s has no nn_module"%type(self).__name__)

    def _sample(self, pos: int | float | None, sr: int):
        if isinstance(pos, float):
            return int(pos * sr)
        return pos

    def load_file(self, path = None, start = None, end = None, duration = None, target_sr = None, channels = None):
        path = path or self.path
        out, sr = load_file(path)

        if out.shape[0] > channels:
            out = out[:channels]
        elif out.shape[0] < channels: 
            out = out[torch.arange(channels)%out.shape[0]]

        # get start
        start = self._sample(start or self.start, sr)
        if end is None and duration is not None: 
            end = duration if start is None else start + duration
        # get end
        end = self._sample(end, sr)

        out = out[..., start:end]
        if target_sr != sr and target_sr is not None:
            out = torchaudio.functional.resample(out, sr, target_sr)
        return out    

    def from_fragment(self, fragment, write: bool = None):
        raise NotImplementedError()

    @classmethod
    def write_gin_config(cls, config_path, config = None):
        if config is None: 
            generate_config_from_obj(cls, config_path, default_feature_pattern)
        else:
            config = parse_config_pattern(config, name=cls.__name__)
            with open(config_path, "w+") as f: 
                f.write(config)

    def extract(self, fragment=None, current_key = None, feature_hash = None):
        if fragment is not None:
            meta = self.from_fragment(fragment)
        else:
            raise FeatureException("fragment must be given")
        if (self.has_hash) and (current_key) is not None and (feature_hash is not None): 
            if self.hash_from_feature is not None:
                meta_hash = self.hash_from_feature(meta)
            else:
                meta_hash = meta
            if meta_hash not in feature_hash[self.feature_name]:
                feature_hash[self.feature_name][meta_hash] = [current_key]
            else:
                feature_hash[self.feature_name][meta_hash].append(current_key)
        return meta

    def read(self, fragment):
        return fragment.get_data(self.feature_name)

    def __call__(self, *args, **kwargs): 
        raise NotImplementedError()

    # features computing batches at once (e.g. on GPU, see AudioDataset.compute_batch_features)
    supports_batch = False

    def batch_call(self, x: torch.Tensor) -> torch.Tensor:
        """computes the feature of a batch of audio (batch, channels, samples), as extracted for each item. Features 
        with supports_batch compute the whole batch at once ; others are computed item by item."""
        if self.supports_batch:
            return torch.as_tensor(self(x))
        return torch.stack([torch.as_tensor(self(x[i])) for i in range(x.shape[0])])
        



def check_feature_configs(module, path):
    feature_class = getattr(module, "AcidsDatasetFeature")
    feature_subclasses = get_subclasses_from_package(module, feature_class, exclude=["beat_this"])
    os.makedirs(path, exist_ok=True)
    for feature in feature_subclasses:
        # private classes (e.g. base classes of features) have no configuration
        if feature == feature_class or feature.__name__.startswith("_"): continue
        gin_config_name = feature.__name__.lower()
        if hasattr(feature, "gin_configs"):
            for config_name, config_str in feature.gin_configs.items(): 
                gin_config_path = (path / (gin_config_name + f"_{config_name}.gin")).resolve()
                feature.write_gin_config(gin_config_path, config_str)
        else:
            gin_config_path = (path / (gin_config_name + ".gin")).resolve()
            if not gin_config_path.exists():
                feature.write_gin_config(gin_config_path)
