import importlib
from types import ModuleType
import dill
import copy
import inspect, pkgutil
import logging
import enum
from contextlib import ContextDecorator
import os
from pathlib import Path
import random
import math
import torch
import torchaudio
import gin

from . import FEATURES_GIN_PATH, TRANSFORM_GIN_PATH

_CACHED_MODULES = {}
_VALID_BACKENDS = ['numpy', 'torch', 'jax']

def load_backend(backend: str):
    try:
        return importlib.import_module(backend)
    except ModuleNotFoundError:
        return None

def get_backend(backend: str):
    if backend not in _VALID_BACKENDS: 
        raise ValueError('backend %s not available.'%backend)
    if backend not in _CACHED_MODULES:
        _CACHED_MODULES[backend] = load_backend(backend)
    return _CACHED_MODULES[backend]

def load_file(file_path):
    return torchaudio.load(file_path)


def copy_for_gin_config_copy(obj):
    if isinstance(obj, list):
        return list(map(copy_for_gin_config_copy, obj))
    elif isinstance(obj, tuple):
        return tuple(map(copy_for_gin_config_copy, obj))
    else:
        if isinstance(obj, gin.config.ConfigurableReference):
            new_obj = gin.config.ConfigurableReference(obj._scoped_selector, obj._evaluate)
            return new_obj
        else:
            return copy.deepcopy(obj)


def copy_gin_config(config):
    copy_config = {}
    for field_ref, field_dict in config.items():
        field_copy = {}
        for k, v in field_dict.items(): 
            field_copy[k] = copy_for_gin_config_copy(v)
        copy_config[field_ref] = field_copy
    return copy_config


class GinEnv(object):
    def __init__(self, paths=[], configs=[], bindings=[], clear:bool = True):
        self._paths = checklist(paths)
        self._configs = checklist(configs)
        self._bindings = checklist(bindings)
        self.keep_constants = True
        self._dict = None
        self._clear = clear 

    def _copy_gin_dict(self):
        gin_dict = {}
        for k, v in gin.config.__dict__.items(): 
            if isinstance(v, ModuleType) or callable(v): continue
            try:
                if k == "_SCOPE_MANAGER":
                    gin_dict[k] = {'active_scopes': v.active_scopes,
                                   'current_scope': v.current_scope,
                                   '_active_scopes': v._active_scopes}
                elif k == "_CONFIG":
                    gin_dict[k] = copy_gin_config(v)
                else:
                    gin_dict[k] = copy.deepcopy(v)
            except: 
                try:
                    gin_dict[k] = dill.dumps(v)
                except: 
                    gin_dict[k] = v
        return gin_dict
    
    def __enter__(self):
        self._dict = self._copy_gin_dict()
        if self._clear:
            gin.clear_config()
        if self.keep_constants: 
            gin.config._CONSTANTS = self._dict['_CONSTANTS']
        for p in self._paths: 
            gin.add_config_file_search_path(p)
        if len(self._configs) or len(self._bindings):
            gin.parse_config_files_and_bindings(self._configs, self._bindings)
        gin.unlock_config()

    def __exit__(self, *args):
        gin.clear_config()
        scope_manager = gin.config._ScopeManager()
        scope_manager.__dict__.update(self._dict['_SCOPE_MANAGER'])
        self._dict['_SCOPE_MANAGER'] = scope_manager
        for k, v in self._dict.items():
            if isinstance(v, bytes):
                self._dict[k] = dill.loads(v)
        gin.config.__dict__.update(self._dict)


def loudness(waveform: torch.Tensor, sample_rate: int):
    r"""
    Custom extension of torchaudio loudness, allowing loudness computation for small chunks
    """

    if not torch.is_tensor(waveform):
        waveform = torch.from_numpy(waveform)

    if waveform.size(-2) > 5:
        raise ValueError("Only up to 5 channels are supported.")

    gate_duration = min(0.4, waveform.size(-1) / sample_rate)
    overlap = 0.75
    gamma_abs = -70.0
    kweight_bias = -0.691
    gate_samples = int(round(gate_duration * sample_rate))
    step = int(round(gate_samples * (1 - overlap)))

    # Apply K-weighting
    waveform = torchaudio.functional.treble_biquad(waveform, sample_rate, 4.0, 1500.0, 1 / math.sqrt(2))
    waveform = torchaudio.functional.highpass_biquad(waveform, sample_rate, 38.0, 0.5)

    # Compute the energy for each block
    energy = torch.square(waveform).unfold(-1, gate_samples, step)
    energy = torch.mean(energy, dim=-1)

    # Compute channel-weighted summation
    g = torch.tensor([1.0, 1.0, 1.0, 1.41, 1.41], dtype=waveform.dtype, device=waveform.device)
    g = g[: energy.size(-2)]

    energy_weighted = torch.sum(g.unsqueeze(-1) * energy, dim=-2)
    loudness = -0.691 + 10 * torch.log10(energy_weighted)

    # Apply absolute gating of the blocks
    gated_blocks = loudness > gamma_abs
    gated_blocks = gated_blocks.unsqueeze(-2)

    energy_filtered = torch.sum(gated_blocks * energy, dim=-1) / torch.count_nonzero(gated_blocks, dim=-1)
    energy_weighted = torch.sum(g * energy_filtered, dim=-1)
    gamma_rel = kweight_bias + 10 * torch.log10(energy_weighted) - 10

    # Apply relative gating of the blocks
    gated_blocks = torch.logical_and(gated_blocks.squeeze(-2), loudness > gamma_rel.unsqueeze(-1))
    gated_blocks = gated_blocks.unsqueeze(-2)

    energy_filtered = torch.sum(gated_blocks * energy, dim=-1) / torch.count_nonzero(gated_blocks, dim=-1)
    energy_weighted = torch.sum(g * energy_filtered, dim=-1)
    if energy_weighted.isnan():
        return torch.tensor(-torch.inf)
    LKFS = kweight_bias + 10 * torch.log10(energy_weighted)
    return LKFS

def checklist(item, n=1, copy=False):
    """Repeat list elemnts
    """
    if not isinstance(item, (list, )):
        if copy:
            item = [copy.deepcopy(item) for _ in range(n)]
        elif isinstance(item, torch.Size):
            item = [i for i in item]
        else:
            item = [item]*n
    return item

def get_random_hash(n=8):
    return "".join([chr(random.randrange(97,122)) for i in range(n)])


@gin.configurable(module="features")
def parse_features(features=None, device=None, add_args=None):
    if features is None: 
        return []
    else:
        f = checklist(features)
        add_args = checklist(add_args, n=len(f))
        for i, a in enumerate(add_args): 
            if a is None: 
                add_args[i] = (tuple(), dict())
        if device is not None: 
            set_gin_constant("DEVICE", device)
        return [f(*add_args[i][0], **add_args[i][1]) for i, f in enumerate(checklist(features))]



def feature_from_gin_config(config_path, add_args=None):
    config_path = checklist(config_path)
    for i in range(len(config_path)):
        if not os.path.splitext(config_path[i])[1] == ".gin": config_path[i] += ".gin"
    with GinEnv(configs=config_path, paths=FEATURES_GIN_PATH, clear=False):
        feature = parse_features(add_args=add_args)
    return feature



@gin.configurable(module="transforms")
def parse_transform(transform, add_args=None):
    return transform(*add_args[0], **add_args[1])


def transform_from_gin_config(config_path, add_args=None):
    config_path = checklist(config_path)
    if add_args is None: 
        add_args = (tuple(), dict())
    for i in range(len(config_path)):
        if not os.path.splitext(config_path[i])[1] == ".gin": config_path[i] += ".gin"
    with GinEnv(configs=config_path, paths=TRANSFORM_GIN_PATH, clear=False):
        transform = parse_transform(add_args=add_args)
    return transform


def get_available_cuda_device():
    #TODO (or not, let user decide with CUDA_VISIBLE_DEVICES)
    return torch.device('cuda')


def get_default_accelerated_device():
    cuda_available = os.environ.get('USE_CUDA', True) and torch.cuda.is_available()
    if cuda_available:
        if os.environ.get('CUDA_AVAILABLE_DEVICES'):
            get_available_cuda_device()
        else:
            return torch.device('cuda')
    mps_available = os.environ.get('USE_MPS', False) and torch.mps.is_available()
    if mps_available: 
        return torch.device('mps')
    return torch.device('cpu')

def get_accelerated_device(accelerator = None):
    if accelerator == "cpu":
        return torch.device("cpu")
    elif accelerator == "cuda":
        return torch.device('cpu') if not torch.cuda.is_available() else get_available_cuda_device()
    elif accelerator == "mps":
        return torch.device('cpu') if not torch.mps.is_available() else torch.device('mps')
    else:
        raise ValueError('accelerator value %s not handled.'%accelerator)


class PadMode(enum.Enum):
    DISCARD = 0
    ZERO = 1
    REPEAT = 2
    REPLICATE = 3
    REFLECT = 4


def pad(
        chunk,
        target_size,
        pad_mode,
        discard_if_lower_than: None | float | int = None
    ):
    if isinstance(pad_mode, str):
        pad_mode = getattr(PadMode, pad_mode.upper())
    if pad_mode == PadMode.DISCARD:
        return None
    if discard_if_lower_than is not None:
        if isinstance(discard_if_lower_than, int):
            if chunk.shape[-1] < discard_if_lower_than: 
                return None
        elif isinstance(discard_if_lower_than, float):
            if chunk.shape[-1] < int(discard_if_lower_than * target_size): 
                return None
        else: 
            raise TypeError('got type %s for discard_if_lower_than'%type(discard_if_lower_than))
    if pad_mode == PadMode.ZERO:
        return torch.nn.functional.pad(chunk, (0, target_size - chunk.shape[-1]), mode="constant", value=0.)
    elif pad_mode == PadMode.REPEAT:
        n_iter = 0
        while chunk.shape[-1] < target_size:
            if n_iter > 1: 
                logging.warning('applying repeat pad more than once ; may provoke undesired behaviour.')
            chunk = torch.nn.functional.pad(chunk, (0, min(target_size - chunk.shape[-1], chunk.shape[-1] - 1)), mode="circular")
            n_iter += 1
        return chunk
    elif pad_mode == PadMode.REPLICATE:
        return torch.nn.functional.pad(chunk, (0, target_size - chunk.shape[-1]), mode="replicate")
    elif pad_mode == PadMode.REFLECT:
        n_iter = 0
        while chunk.shape[-1] < target_size:
            chunk = torch.nn.functional.pad(chunk, (0, min(target_size - chunk.shape[-1], chunk.shape[-1] - 1)), mode="reflect")
            n_iter += 1
        if n_iter > 1: 
            logging.warning('applied reflect pad more than once ; may provoke undesired behaviour.')
        return chunk
    else:
        raise ValueError('pad mode %s not recognized.'%pad_mode)

def mirror_pad(tensor, target_size, mode="reflect", value=0):
    if (tensor.shape[-1] < target_size):
        pad_len = target_size - tensor.shape[-1]
        pad_bef, pad_aft = math.floor(pad_len / 2), math.ceil(pad_len / 2)
        return torch.nn.functional.pad(tensor, (pad_bef, pad_aft), mode=mode, value=value)
    elif(tensor.shape[-1] > target_size): 
        crop_len = target_size - tensor.shape[-1]
        crop_bef, crop_aft = math.floor(crop_len / 2), math.ceil(crop_len / 2)
        return tensor[..., crop_bef:-crop_aft]
    else:
        return tensor


def match_loudness(signal, target_signal, sr):
    if signal.ndim == 3: 
        return torch.stack(0, map(match_loudness, signal))
    current_loudness = torchaudio.functional.loudness(signal, sr)
    target_loudness = torchaudio.functional.loudness(target_signal, sr)
    return torchaudio.functional.gain(signal, target_loudness - current_loudness)
    

def walk_modules(package):
    yield package
    if hasattr(package, '__path__'):
        for _, name, ispkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            submod = importlib.import_module(name)
            yield from walk_modules(submod)


def get_subclasses_from_package(package, filter_class, exclude = []):
    transform_list = []
    for mod in walk_modules(package):
        if mod.__name__ in exclude: pass
        for name, obj in inspect.getmembers(mod):
            if isinstance(obj, type):
                if issubclass(obj, filter_class) and (obj not in transform_list):
                    transform_list.append(obj)
    return transform_list


def set_gin_constant(constant, value, replace: bool = True):
    if constant in gin.config._CONSTANTS:
        if replace:
            gin.config._CONSTANTS[constant] = value
    else:
        gin.constant(constant, value)


def parse_config_pattern(pattern: str, **kwargs):
    for kwname, kwvalue in kwargs.items():
        kwname = kwname.upper()
        pattern = pattern.replace("{{%s}}"%kwname, kwvalue)
    return pattern

def generate_config_from_obj(transform_class, config_path, pattern):
    gin_args = []
    transform_args = transform_class.init_signature()
    for param_name, param in transform_args.items():
        if param_name in transform_class.dont_export_to_gin_config: continue
        if param_name == "sr": 
            gin_args.append("sr = %SAMPLE_RATE")
        else:
            default = param._default
            if (param._default == inspect._empty): 
                continue
            if isinstance(param._default, str): 
                default = f"\"{default}\""
            gin_args.append(f"{param_name} = {default}")
    gin_args = "\n\t".join(gin_args)
    gin_out = parse_config_pattern(pattern, name=transform_class.__name__, args=gin_args)
    with open(config_path, "w+") as f: 
        f.write(gin_out)

def apply_nested(func, nested):
    if isinstance(nested, list) or isinstance(nested, tuple):
        return type(nested)(apply_nested(func, x) for x in nested)
    else:
        return func(nested)
    
def get_folder_size(path):
    size = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
    size /= (1024 ** 3)
    return size
  
