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
import torch.nn.functional as F
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

def audio_info(file_path):
    """returns header information (sr, channels, n_frames, duration, codec) without decoding the file. 
    Uses soundfile (exact frame counts), and torchcodec for formats unsupported by libsndfile."""
    file_path = str(file_path)
    try:
        import soundfile as sf
        info = sf.info(file_path)
        return {
            'sr': int(info.samplerate), 
            'channels': int(info.channels),
            'n_frames': int(info.frames), 
            'duration': float(info.frames / info.samplerate),
            'codec': info.subtype,
            'backend': 'soundfile'
        }
    except Exception as sf_exception:
        try:
            from torchcodec.decoders import AudioDecoder
        except ImportError:
            raise sf_exception
        metadata = AudioDecoder(file_path).metadata
        duration = metadata.duration_seconds_from_header or metadata.duration_seconds
        return {
            'sr': int(metadata.sample_rate), 
            'channels': int(metadata.num_channels), 
            'n_frames': int(round(duration * metadata.sample_rate)),
            'duration': float(duration),
            'codec': metadata.codec,
            'backend': 'torchcodec'
        }

def load_audio_range(file_path, frame_offset: int = 0, num_frames: int = -1, backend: str | None = None):
    """loads num_frames samples of a file starting at frame_offset (in original sampling rate) without 
    decoding the whole file. Returns a (channels, time) float32 tensor and the sampling rate.

    backend can be "soundfile" (sample-accurate seeking), "torchcodec" (FFmpeg-based seeking, for 
    formats unsupported by libsndfile), or None to try soundfile first."""
    file_path = str(file_path)
    if backend in [None, "soundfile"]:
        try:
            import soundfile as sf
            with sf.SoundFile(file_path) as f:
                sr = f.samplerate
                f.seek(min(frame_offset, f.frames))
                data = f.read(num_frames, dtype="float32", always_2d=True)
            return torch.from_numpy(data.T.copy()), sr
        except Exception as e:
            if backend == "soundfile": raise e
    from torchcodec.decoders import AudioDecoder
    decoder = AudioDecoder(file_path)
    sr = decoder.metadata.sample_rate
    start = frame_offset / sr
    stop = None if num_frames < 0 else (frame_offset + num_frames) / sr
    samples = decoder.get_samples_played_in_range(start, stop)
    data = samples.data
    # first decoded frame may start slightly before the requested position
    offset = max(0, round((start - samples.pts_seconds) * sr))
    data = data[..., offset:]
    if num_frames >= 0:
        data = data[..., :num_frames]
    return data.float(), sr


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
    def __init__(self, paths=[], configs=[], bindings=[], clear:bool = True, import_constants: bool = True, keep_constants: bool = False):
        self._paths = checklist(paths)
        self._configs = checklist(configs)
        self._bindings = checklist(bindings)
        self.keep_constants = keep_constants
        self.import_constants = import_constants
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
        return copy.deepcopy(gin_dict)
    
    def __enter__(self):
        self._dict = self._copy_gin_dict()
        if self._clear:
            gin.clear_config()
        if self.import_constants: 
            gin.config._CONSTANTS = copy.copy(self._dict['_CONSTANTS'])
        for p in self._paths: 
            gin.add_config_file_search_path(p)
        if len(self._configs) or len(self._bindings):
            gin.parse_config_files_and_bindings(self._configs, self._bindings)
        gin.unlock_config()

    def __exit__(self, *args):
        constants = gin.config._CONSTANTS.copy()
        gin.clear_config(clear_constants=True)
        scope_manager = gin.config._ScopeManager()
        scope_manager.__dict__.update(self._dict['_SCOPE_MANAGER'])
        self._dict['_SCOPE_MANAGER'] = scope_manager
        for k, v in self._dict.items():
            if isinstance(v, bytes):
                self._dict[k] = dill.loads(v)
        if self.keep_constants: 
            for k, v in constants._selector_map.items():
                self._dict['_CONSTANTS']._selector_map[k] = v
            for k, v in constants._selector_tree.items():
                self._dict['_CONSTANTS']._selector_tree[k] = dict(v)
        gin.config.__dict__.update(self._dict)


def _k_weighting_coefficients(sr: float):
    """(b, a) coefficients of the two biquads of K-weighting (ITU-R BS.1770) : high shelf (+4 dB at 1500 Hz) and 
    high-pass (38 Hz), as in torchaudio treble_biquad and highpass_biquad."""
    gain, cf, Q = 4.0, 1500.0, 1.0 / math.sqrt(2.0)
    w0 = 2.0 * math.pi * cf / sr
    A = 10.0 ** (gain / 40.0)
    alpha = math.sin(w0) / (2.0 * Q)
    shelf_b = [A * ((A + 1) + (A - 1) * math.cos(w0) + 2.0 * math.sqrt(A) * alpha), 
               -2.0 * A * ((A - 1) + (A + 1) * math.cos(w0)), 
               A * ((A + 1) + (A - 1) * math.cos(w0) - 2.0 * math.sqrt(A) * alpha)]
    shelf_a = [(A + 1) - (A - 1) * math.cos(w0) + 2.0 * math.sqrt(A) * alpha, 
               2.0 * ((A - 1) - (A + 1) * math.cos(w0)), 
               (A + 1) - (A - 1) * math.cos(w0) - 2.0 * math.sqrt(A) * alpha]
    w0 = 2.0 * math.pi * 38.0 / sr
    alpha = math.sin(w0) / (2.0 * 0.5)
    hp_b = [(1.0 + math.cos(w0)) / 2.0, -(1.0 + math.cos(w0)), (1.0 + math.cos(w0)) / 2.0]
    hp_a = [1.0 + alpha, -2.0 * math.cos(w0), 1.0 - alpha]
    normalize = lambda b, a: ([x / a[0] for x in b], [x / a[0] for x in a])
    return [normalize(shelf_b, shelf_a), normalize(hp_b, hp_a)]


LOUDNESS_MODES = ["lufs", "db", "dba"]


def a_weighting(freqs: torch.Tensor) -> torch.Tensor:
    """A-weighting power gains (IEC 61672, 0 dB at 1 kHz) of frequencies in Hz."""
    f2 = freqs.square()
    ra = (12194. ** 2) * f2.square() / ((f2 + 20.6 ** 2) * torch.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194. ** 2))
    return (ra * 10 ** (2. / 20)).square()


def mean_square(frames: torch.Tensor, sample_rate: int, a_weighted: bool = False) -> torch.Tensor:
    """mean square of signals (..., n) -> (...), A-weighted with a_weighted (computed from the weighted spectrum of
    rectangular frames : without weighting, it equals the mean square of the signal, by Parseval's theorem)."""
    if not a_weighted:
        return frames.square().mean(-1)
    n = frames.shape[-1]
    spectrum = torch.fft.rfft(frames, dim=-1).abs().square()
    weights = a_weighting(torch.fft.rfftfreq(n, d=1. / sample_rate).to(frames.device, frames.dtype))
    weights[1:(n + 1) // 2] *= 2
    return (spectrum * weights).sum(-1) / n ** 2


def rms_level(waveform: torch.Tensor, sample_rate: int, frame_length: int | None = None, keep_channels: bool = False,
              a_weighted: bool = False) -> torch.Tensor:
    """RMS level in dBFS (or A-weighted, in dB(A), with a_weighted) of waveforms (..., channels, samples), with the
    same frames and output shapes as loudness : (...), (..., num_frames) with frame_length (non-overlapping frames,
    the last incomplete frame is discarded), (..., channels) with keep_channels, (..., channels, num_frames) with both.
    Without keep_channels, levels are the ones of the mix (mean) of channels. Silence gives -inf."""
    if not torch.is_tensor(waveform):
        waveform = torch.from_numpy(waveform)
    x = waveform
    if frame_length is not None:
        x = x[..., :frame_length * (x.shape[-1] // frame_length)].unfold(-1, frame_length, frame_length)
    else:
        x = x.unsqueeze(-2)
    if not keep_channels:
        x = x.mean(-3)
    level = 10 * torch.log10(mean_square(x, sample_rate, a_weighted=a_weighted))
    return level if frame_length is not None else level.squeeze(-1)


def level(waveform: torch.Tensor, sample_rate: int, mode: str = "lufs", frame_length: int | None = None, keep_channels: bool = False) -> torch.Tensor:
    """level of waveforms (..., channels, samples) : loudness in LUFS (mode="lufs", see loudness), RMS level in dBFS 
    ("db") or A-weighted RMS level in dB(A) ("dba", see rms_level). Silence gives -inf."""
    if mode == "lufs":
        return loudness(waveform, sample_rate, frame_length=frame_length, keep_channels=keep_channels)
    if mode in ["db", "dba"]:
        return rms_level(waveform, sample_rate, frame_length=frame_length, keep_channels=keep_channels, a_weighted=mode == "dba")
    raise ValueError("mode must be in %s, got %s"%(LOUDNESS_MODES, mode))


def loudness(waveform: torch.Tensor, sample_rate: int, frame_length: int | None = None, keep_channels: bool = False):
    r"""
    Custom extension of torchaudio loudness, allowing loudness computation for small chunks.

    Args:
        waveform: input tensor of shape (..., channels, samples)
        sample_rate: sample rate
        frame_length: if given, split audio into non-overlapping frames of this length (in samples),
                      discarding the last incomplete frame, and return per-frame loudness
        keep_channels: if True, compute loudness per channel instead of combined LUFS

    Returns:
        Loudness tensor. Shape depends on arguments:
          default:                   (...,)
          frame_length only:         (..., num_frames)
          keep_channels only:        (..., channels)
          frame_length+keep_channels: (..., channels, num_frames)
    """
    if not torch.is_tensor(waveform):
        waveform = torch.from_numpy(waveform)

    C = waveform.size(-2)
    if C > 5:
        raise ValueError("Only up to 5 channels are supported.")

    batch_shape = waveform.shape[:-2]
    T = waveform.size(-1)
    B = math.prod(batch_shape)  # 1 when there are no batch dims

    # --- Frame splitting: crop to multiple of frame_length, unfold into frames ---
    if frame_length is not None:
        remainder = T % frame_length
        if remainder != 0:
            # last incomplete frame is discarded (as in features.loudness.LoudnessScript)
            waveform = waveform[..., :frame_length * (T // frame_length)]
        # (..., C, T) -> (..., C, num_frames, frame_length)
        waveform = waveform.unfold(-1, frame_length, frame_length)
        num_frames = waveform.size(-2)
        # permute C and num_frames: (..., num_frames, C, frame_length)
        ndim = waveform.ndim
        perm = list(range(ndim - 3)) + [ndim - 2, ndim - 3, ndim - 1]
        waveform = waveform.permute(*perm).contiguous()
        # flatten batch and frames -> (B*num_frames, C, frame_length)
        waveform = waveform.reshape(B * num_frames, C, frame_length)
    else:
        num_frames = None
        waveform = waveform.reshape(B, C, T)

    # --- K-weighting ---
    # filtered without clamping (torchaudio biquads clamp outputs to [-1, 1], that biases the loudness of loud signals)
    for b, a in _k_weighting_coefficients(sample_rate):
        waveform = torchaudio.functional.lfilter(waveform, torch.tensor(a, dtype=waveform.dtype, device=waveform.device), 
                                                 torch.tensor(b, dtype=waveform.dtype, device=waveform.device), clamp=False)

    gate_duration = min(0.4, waveform.size(-1) / sample_rate)
    overlap = 0.75
    gamma_abs = -70.0
    kweight_bias = -0.691
    gate_samples = int(round(gate_duration * sample_rate))
    step = int(round(gate_samples * (1 - overlap)))

    # energy: (B_eff, C, W) where W = num_gate_windows
    energy = torch.square(waveform).unfold(-1, gate_samples, step).mean(dim=-1)

    g = torch.tensor([1.0, 1.0, 1.0, 1.41, 1.41], dtype=waveform.dtype, device=waveform.device)[:C]

    if keep_channels:
        # Per-channel path: all ops over window dim W, preserve C
        # lval: (B_eff, C, W)
        lval = kweight_bias + 10 * torch.log10(energy.clamp(min=1e-10))

        # Absolute gating per channel
        gated = lval > gamma_abs  # (B_eff, C, W)
        count = gated.sum(dim=-1).clamp(min=1)
        energy_filtered = (gated * energy).sum(dim=-1) / count  # (B_eff, C)
        gamma_rel = kweight_bias + 10 * torch.log10(energy_filtered.clamp(min=1e-10)) - 10  # (B_eff, C)

        # Relative gating per channel
        gated_rel = gated & (lval > gamma_rel.unsqueeze(-1))  # (B_eff, C, W)
        count_rel = gated_rel.sum(dim=-1).clamp(min=1)
        energy_final = (gated_rel * energy).sum(dim=-1) / count_rel  # (B_eff, C)

        LKFS = torch.where(
            gated_rel.any(dim=-1),
            kweight_bias + 10 * torch.log10(energy_final.clamp(min=1e-10)),
            torch.full_like(energy_final, -torch.inf),
        )  # (B_eff, C)

        if num_frames is not None:
            LKFS = LKFS.reshape(*batch_shape, num_frames, C).transpose(-2, -1)
        else:
            LKFS = LKFS.reshape(*batch_shape, C)
    else:
        # Combined-channel path: reduce C via g weights first
        # energy_w: (B_eff, W)
        energy_w = torch.einsum('bcw,c->bw', energy, g)
        lval = kweight_bias + 10 * torch.log10(energy_w.clamp(min=1e-10))

        # Absolute gating
        gated = lval > gamma_abs  # (B_eff, W)
        count = gated.sum(dim=-1).clamp(min=1)
        energy_filtered = torch.einsum('bcw,bw->bc', energy, gated.float()) / count.unsqueeze(-1)  # (B_eff, C)
        energy_w_f = torch.einsum('bc,c->b', energy_filtered, g)  # (B_eff,)
        gamma_rel = kweight_bias + 10 * torch.log10(energy_w_f.clamp(min=1e-10)) - 10

        # Relative gating
        gated_rel = gated & (lval > gamma_rel.unsqueeze(-1))  # (B_eff, W)
        count_rel = gated_rel.sum(dim=-1).clamp(min=1)
        energy_filtered = torch.einsum('bcw,bw->bc', energy, gated_rel.float()) / count_rel.unsqueeze(-1)
        energy_w_f = torch.einsum('bc,c->b', energy_filtered, g)

        LKFS = torch.where(
            gated_rel.any(dim=-1),
            kweight_bias + 10 * torch.log10(energy_w_f.clamp(min=1e-10)),
            torch.full_like(energy_w_f, -torch.inf),
        )  # (B_eff,)

        if len(batch_shape) > 0:
            if num_frames is not None:
                LKFS = LKFS.reshape(*batch_shape, num_frames)
            else:
                LKFS = LKFS.reshape(*batch_shape)
        else: 
            LKFS = LKFS.squeeze(0)


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


def env_flag(name: str, default: bool) -> bool:
    """boolean value of an environment variable ("0", "false", "no", "off" and "" are False), default if not set."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off", "")


def get_default_accelerated_device():
    """accelerated device used with device="auto" : CUDA if available (unless USE_CUDA=0), MPS if available and 
    USE_MPS=1, and CPU otherwise. CUDA devices can be selected with CUDA_VISIBLE_DEVICES."""
    if env_flag('USE_CUDA', True) and torch.cuda.is_available():
        return get_available_cuda_device()
    if env_flag('USE_MPS', False) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def resolve_device(device=None) -> torch.device:
    """device of features : CPU if None, the default accelerated device if "auto" (see get_default_accelerated_device)."""
    if device is None:
        return torch.device('cpu')
    if isinstance(device, str) and device == "auto":
        return get_default_accelerated_device()
    return torch.device(device)

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
    # degenerate chunks : nothing to repeat for empty chunks, and repeat / reflect cannot grow single samples
    if chunk.shape[-1] == 0:
        pad_mode = PadMode.ZERO
    elif chunk.shape[-1] == 1 and pad_mode in [PadMode.REPEAT, PadMode.REFLECT]:
        pad_mode = PadMode.REPLICATE
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


import numpy as np
def frame(
    x: torch.Tensor,
    *,
    frame_length: int,
    hop_length: int,
    axis: int = -1,
) -> np.ndarray:
 
    # x = np.array(x, copy=False, subok=subok)

    if x.shape[axis] < frame_length:
        raise ValueError(
            f"Input is too short (n={x.shape[axis]:d}) for frame_length={frame_length:d}"
        )

    if hop_length < 1:
        raise ValueError(f"Invalid hop_length: {hop_length:d}")

    # put our new within-frame axis at the end for now
    out_strides = x.stride() + tuple([x.stride()[axis]])

    # Reduce the shape on the framing axis
    x_shape_trimmed = list(x.shape)
    x_shape_trimmed[axis] -= frame_length - 1

    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = x.as_strided(out_shape, out_strides)

    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1

    xw = torch.moveaxis(xw, -1, target_axis)

    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    return xw[tuple(slices)]

def frame_with_pad(x: torch.Tensor, frame_length: int, hop_size: int) -> torch.Tensor:
    """
    Extends librosa.util.frame with end padding if required, similar to 
    tf.signal.frame(pad_end=True).

    Returns:
        framed_audio: tensor with shape (n_windows, AUDIO_N_SAMPLES)
    """
    n_frames = math.ceil(x.shape[-1] / hop_size)
    n_pads = (n_frames - 1) * hop_size + frame_length - x.shape[-1]
    x = pad(x, x.size(-1) + n_pads, PadMode.REFLECT)
    # framed_audio = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_size)
    framed_audio = frame(x, frame_length=frame_length, hop_length=hop_size)
    return framed_audio


def match_loudness(signal, target_signal, sr):
    if signal.ndim == 3: 
        return torch.stack([match_loudness(s, t, sr) for s, t in zip(signal, target_signal)])
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
                # generic aliases (e.g. list[int]) are instances of type in Python 3.10, but not classes
                try:
                    is_subclass = issubclass(obj, filter_class)
                except TypeError:
                    continue
                if is_subclass and (obj not in transform_list):
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


def nearest_interp_np(signal: np.ndarray, target_length: int):
    x_new = np.linspace(0, signal.shape[-1] - 1, target_length)
    out = signal[..., np.round(x_new).astype(int)]
    return out

  
