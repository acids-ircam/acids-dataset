import torch, gin.torch
import math
import numpy as np
import re
import librosa
from scipy.interpolate import interp1d
from .base import AcidsDatasetFeature
from . import RAISE_EXC_IF_FEATURE_ERROR
from ..utils import apply_nested, pad, PadMode, nearest_interp_np
from typing import Optional, Callable
from collections import Counter

_AD_F0_METHODS = ["yin", "pyin"] 

yin_config = """
{{NAME}}:
\tmode="yin"
\tfmin = "C2"
\tfmax = "C7"
\tframe_length = 2048
\ttrough_threshold=0.1
\tcenter=True
\tpad_mode="constant"
"""

pyin_config = """
{{NAME}}:
\tmode="pyin"
\tfmin = "C2"
\tfmax = "C7"
\tframe_length = 2048
\tn_thresholds=100
\tbeta_parameters=(2, 18)
\tboltzmann_parameter=2
\tresolution=0.1
\tmax_transition_rate=35.92
\tswitch_prob=0.01
\tno_trough_prob=0.01
\tcenter=True
\tpad_mode="constant"
"""

_post_config = """
features.parse_features:
    features = @features.{{NAME}}
"""

_configs = {
    "yin": f"{yin_config}\n\n{_post_config}", 
    "pyin": f"{pyin_config}\n\n{_post_config}", 
}

try:
    import torchcrepe 
    _AD_F0_METHODS.append("crepe")
    crepe_config = "\n\t".join(['{{NAME}}:', 'mode="crepe"', "fmin=50", "fmax=2006", 'decoder="viterbi"']) 
    _configs['crepe'] = f"{crepe_config}\n\n{_post_config}" 
except ModuleNotFoundError: 
    pass

try:
    import pesto 
    _AD_F0_METHODS.append("pesto")
    pesto_config = "\n\t".join(['{{NAME}}:', 'mode="pesto"']) 
    _configs['pesto'] = f"{pesto_config}\n\n{_post_config}" 
except ModuleNotFoundError: 
    pass

try: 
    import pyworld
    _AD_F0_METHODS.append("world")
    pw_config = "\n\t".join(['{{NAME}}:', 'mode="world"']) 
    _configs['world'] = f"{pw_config}\n\n{_post_config}" 
except ModuleNotFoundError:
    pass

try: 
    import parselmouth
    _AD_F0_METHODS.append("praat")
    praat_config = "\n\t".join(['{{NAME}}:', 'mode="praat"'])
    _configs['praat'] = f"{praat_config}\n\n{_post_config}" 
except ModuleNotFoundError:
    pass


def _reframe_pitch_out(f0_array: np.ndarray | torch.Tensor, audio_array: np.ndarray | torch.Tensor, frame_length: int, win_length: int, hop_length: int) -> torch.Tensor | np.ndarray:
    target_size = math.floor((audio_array.shape[-1] - frame_length) / hop_length) + 1
    if torch.is_tensor(f0_array):
        return torch.nn.functional.interpolate(f0_array.unsqueeze(0), target_size, mode="nearest")[0]
    elif isinstance(f0_array, np.ndarray):
        return nearest_interp_np(f0_array, target_size)
    else:
        raise TypeError(type(f0_array))


@gin.configurable(module="features")
class F0(AcidsDatasetFeature):
    _valid_modes_ = _AD_F0_METHODS
    gin_configs = _configs
    default_frame_length = 512
    def __init__(
            self, 
            mode="pyin",
            sr: int = 44100, 
            name: Optional[str] = None,
            hash_from_feature: Optional[Callable] = None, 
            device: torch.device | None = None, 
            metadata = {},
            **kwargs
    ):
        super().__init__(name=name, hash_from_feature=hash_from_feature, device=device, metadata=metadata)
        self.sr = sr
        assert self.sr is not None, "F0 needs sr keyword"
        self.mode = mode
        self.kwargs = kwargs

    def __repr__(self):
        return "F0(mode=%s, sr=%d)"%(self.mode, self.sr)

    @property
    def default_feature_name(self):
        return "f0"

    @classmethod
    def predict(cls, audio: np.ndarray, mode: str, sr: int, device=None, **kwargs):
        # ["yin", "pyin", "crepe", "pesto"]
        #TODO if channels > 2?  
        fmin = kwargs.get('fmin', 'C2')
        if isinstance(fmin, str): fmin = librosa.note_to_hz(fmin)
        fmax = kwargs.get('fmax', 'C7')
        if isinstance(fmax, str): fmax = librosa.note_to_hz(fmax)
        
        assert mode in _AD_F0_METHODS, "mode %s not available. Available modes : %s"%_AD_F0_METHODS
        if mode in ["yin", "pyin"]:
            add_kwargs = {
                'fmin': fmin,
                'fmax': fmax,
                'frame_length': kwargs.get('frame_length', cls.default_frame_length) or cls.default_frame_length,
                'hop_length': kwargs.get('hop_length', kwargs.get('frame_length', cls.default_frame_length)) or cls.default_frame_length
            }
            f_min = add_kwargs.get('fmin')
            if isinstance(f_min, str): add_kwargs['fmin'] = librosa.note_to_hz(f_min)
            f_max = add_kwargs.get('fmax')
            if isinstance(f_max, str): add_kwargs['fmax'] = librosa.note_to_hz(f_max)
            if mode == "yin":
                f0 = librosa.yin(audio, sr=sr, **add_kwargs)
                f0 = _reframe_pitch_out(f0_array=f0, audio_array=audio, frame_length=add_kwargs['frame_length'], win_length=add_kwargs['frame_length'], hop_length=add_kwargs['hop_length'])
            elif mode == "pyin":
                f0, voiced_flag, voiced_prob = librosa.pyin(audio, sr=sr, **add_kwargs)
                f0 = _reframe_pitch_out(f0_array=f0, audio_array=audio, frame_length=add_kwargs['frame_length'], win_length=add_kwargs['frame_length'], hop_length=add_kwargs['hop_length'])
                voiced_prob = _reframe_pitch_out(f0_array=voiced_flag, audio_array=audio, frame_length=add_kwargs['frame_length'], win_length=add_kwargs['frame_length'], hop_length=add_kwargs['hop_length'])
                f0 = np.stack([f0, voiced_prob], axis=-2)
        elif mode == "crepe": 
            add_kwargs = {
                "decoder": kwargs.get('decoder', 'viterbi'), 
                "hop_length": kwargs.get('hop_length') or cls.default_frame_length,
                'model': kwargs.get('model', 'full'),
                "device": kwargs.get('device', torch.device('cpu'))
            }
            if add_kwargs.get("decoder") is None: 
                add_kwargs["decoder"] = "viterbi"
            if isinstance(add_kwargs.get("decoder"), str):
                add_kwargs["decoder"] = getattr(torchcrepe.decode, add_kwargs["decoder"])
            batch_size = audio.shape[:-1]
            audio = torch.from_numpy(audio).float().reshape(-1, audio.shape[-1]) # type: ignore
            f0 = torch.cat([
                torchcrepe.predict(a[None], sr, return_harmonicity=False, return_periodicity=False, **add_kwargs).cpu() for a in audio
            ], dim=0)
            f0 = _reframe_pitch_out(f0, 
                                    audio_array=audio,
                                    frame_length=kwargs.get('frame_length') or cls.default_frame_length, 
                                    win_length=torchcrepe.core.WINDOW_SIZE,
                                    hop_length=add_kwargs['hop_length'])
            f0 = f0.reshape(*batch_size, f0.shape[-1]).numpy()
        elif mode == "pesto": 
            add_kwargs = {
                "step_size": (kwargs.get('hop_length') or cls.default_frame_length) / sr,
                'model_name': kwargs.get('model', "mir-1k_g7"),
                'reduction': kwargs.get('reduction', 'alwa'),
                'num_chunks': kwargs.get('num_chunks', 8), 
                'convert_to_freq': kwargs.get('convert_to_freq', True)
            }
            audio = torch.from_numpy(audio).float().reshape(-1, audio.shape[-1]) # type: ignore
            if device is not None: audio = audio.to(device)
            out = pesto.predict(audio, sr, **add_kwargs)
            f0 = torch.cat([out[0][None], out[1], out[2]], dim=0)
        elif mode == "world": 
            f0 = []
            if audio.ndim == 1: audio = audio[None]
            batch_size = audio.shape[:-1]
            audio = audio.reshape(-1, audio.shape[-1]).astype(np.float64)
            frame_length = kwargs.get('frame_length') or cls.default_frame_length
            hop_length = kwargs.get('hop_length') or cls.default_frame_length
            for a in audio:
                _f, t = pyworld.dio(a, sr, f0_floor=fmin, f0_ceil=fmax, frame_period=hop_length / sr * 1000)
                f = pyworld.stonemask(a, _f, t, sr) 
                f0.append(f)
            f0 = np.stack(f0)
            f0 = _reframe_pitch_out(f0, 
                                    audio_array=audio,
                                    frame_length=frame_length,
                                    win_length=frame_length,
                                    hop_length=hop_length)
            f0 = f0.reshape(*batch_size, f0.shape[-1])
        elif mode == "praat":
            sound = parselmouth.Sound(audio, sampling_frequency=sr)
            f0 = sound.to_pitch(**add_kwargs).to_array()
            f0 = np.stack([f0[0]['frequency'], f0[0]['strength']], 0)
        f0 = np.where(np.isnan(f0), 0.0, f0)
        f0 = np.where(np.isinf(f0), 0.0, f0)
        # if kwargs.get('frame_length'): 
        #     target_shape = math.ceil(audio.shape[-1] / kwargs['frame_length'])
        #     if target_shape < audio.shape[-1]: 
        #         f0 = f0[..., :target_shape]
        #     elif target_shape > audio.shape[-1]:
        #         f0 = np.pad(f0, pad_width=f0.shape[-1]-target_shape)
        return f0.astype(np.float32)

    def from_fragment(self, fragment, write: bool = True):
        data = fragment.raw_audio
        try: 
            f0 = self.predict(data, self.mode, self.sr, device=self.device, **self.kwargs)
        except RuntimeError as e:
            if RAISE_EXC_IF_FEATURE_ERROR: raise e
            return  
        if write:
            fragment.put_array(self.feature_name, f0)
        return f0

    def __call__(self, x):
        out = self.predict(x.numpy(), self.mode, self.sr, device=self.device, **self.kwargs)
        out = torch.from_numpy(out).to(x)
        return out


def f0_to_pitch(x):
    if np.allclose(x, 0.):
        x = -1
    else:
        note = librosa.hz_to_note(x)
        try: 
            root = re.match(r"^([A-G]+[#b♯♭]?)(\-?\d+)$", note).groups()[0]
            x = Pitch.idx_hash()[root]
        except Exception: 
            pass
    return x
    

@gin.configurable(module="features")
class Pitch(F0): 
    
    def __repr__(self):
        return "Pitch(mode=%s, sr=%d)"%(self.mode, self.sr)

    @property
    def default_feature_name(self):
        return "pitch"

    @classmethod
    def note_hash(cls):
        return {-1: "X", 0: "A", 1: "A♯", 2:"B", 3:"C", 4:"C♯", 5:"D", 6:"D♯", 7:"E", 8: "F", 9:"F♯", 10:"G", 11:"G♯"}
    @classmethod
    def idx_hash(cls):
        return {v: k for k, v in cls.note_hash().items()}

    @property
    def has_hash(self):
        return True 
    
    def hash_from_feature(self, meta): 
        if self.mode in ["pyin", "pesto"]:
            pitch = Counter(meta[0].flatten().tolist()).most_common(1)[0][0]
        else:
            pitch = Counter(meta.flatten().tolist()).most_common(1)[0][0]
        return Pitch.note_hash()[pitch]

    @staticmethod
    def predict(audio: np.ndarray, mode: str, sr: int, **add_kwargs):
        f0 = F0.predict(audio, mode, sr, **add_kwargs)
        note = apply_nested(f0_to_pitch, f0.tolist())
        return np.array(note)

    def __call__(self, x):
        out = self.predict(x.numpy(), self.mode, self.sr, device=self.device, **self.kwargs)
        out = torch.from_numpy(out).to(x.device)
        return out    

