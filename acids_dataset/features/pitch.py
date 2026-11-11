import torch, gin.torch
import torchaudio
import math
import numpy as np
import re
import librosa
from scipy.interpolate import interp1d
from .base import AcidsDatasetFeature
from . import RAISE_EXC_IF_FEATURE_ERROR
from ..utils import apply_nested, pad, PadMode, nearest_interp_np
from typing import List, Optional, Callable
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
ad.features.parse_features:
    features = @ad.features.{{NAME}}
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


def _min_yin_frame_length(sr: float, fmin: float) -> int:
    """minimal frame length such that two periods of fmin fit in the frame (librosa requires at least one period, and 
    warns of inaccurate pitch detection under two periods)."""
    return int(math.ceil(sr / fmin)) * 2 + 1


def _zero_pad_to(audio: np.ndarray, length: int) -> np.ndarray:
    """pads audio (last axis) up to length with trailing zeros. Zeros do not introduce spurious periodicities, 
    whereas reflect or wrap padding create phase discontinuities that bias pitch estimation."""
    missing = int(length) - audio.shape[-1]
    if missing <= 0:
        return audio
    pad_width = [(0, 0)] * (audio.ndim - 1) + [(0, missing)]
    return np.pad(audio, pad_width, mode="constant")


def _align_frames(values: np.ndarray, audio_length: int, frame_length: int, hop_length: int, analysis_length: int, center: bool) -> np.ndarray:
    """selects the analysis frames (last axis) whose centers match the output frames : output frame j covers samples 
    [j * hop, j * hop + frame_length], and analysis frame k is centered on k * hop (center=True), or on 
    k * hop + analysis_length / 2 (center=False)."""
    n_target = max(1, (audio_length - frame_length) // hop_length + 1)
    offset = frame_length / (2 * hop_length) if center else (frame_length - analysis_length) / (2 * hop_length)
    # halves are rounded up (np.round rounds to even), towards frames inside the signal
    index = np.clip(np.floor(np.arange(n_target) + offset + 0.5).astype(int), 0, values.shape[-1] - 1)
    return values[..., index]


def _reframe_pitch_out(f0_array: np.ndarray | torch.Tensor, audio_array: np.ndarray | torch.Tensor, frame_length: int, win_length: int, hop_length: int) -> torch.Tensor | np.ndarray:
    target_size = math.floor((audio_array.shape[-1] - frame_length) / hop_length) + 1
    if torch.is_tensor(f0_array):
        return torch.nn.functional.interpolate(f0_array.unsqueeze(0), target_size, mode="nearest")[0]
    elif isinstance(f0_array, np.ndarray):
        return nearest_interp_np(f0_array, target_size)
    else:
        raise TypeError(type(f0_array))


@gin.configurable(module="ad.features")
class F0(AcidsDatasetFeature):
    _valid_modes_ = _AD_F0_METHODS
    supports_batch = True
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
        self.fmin = kwargs.get('fmin', 'C2')
        if isinstance(self.fmin, str): self.fmin = librosa.note_to_hz(self.fmin)
        self.fmax = kwargs.get('fmax', 'C7')
        if isinstance(self.fmax, str): self.fmax = librosa.note_to_hz(self.fmax)
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
        
        assert mode in _AD_F0_METHODS, "mode %s not available. Available modes : %s"%(mode, _AD_F0_METHODS)
        # inputs shorter than a frame are padded to a frame
        audio = _zero_pad_to(audio, kwargs.get('frame_length') or cls.default_frame_length)
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
            # periods of fmin must fit in analysis frames : if frame_length is too small, frames are analyzed with a 
            # larger window (frames remain centered with the same hop_length, such that output framing is unchanged)
            analysis_kwargs = {
                **add_kwargs, 
                'frame_length': max(add_kwargs['frame_length'], _min_yin_frame_length(sr, add_kwargs['fmin'])),
                'center': kwargs.get('center', True),
                'pad_mode': kwargs.get('pad_mode', 'constant')
            }
            # librosa frames are centered (with padding) : output frames take the analysis frames with the same centers
            align = lambda x: _align_frames(np.asarray(x), audio.shape[-1], add_kwargs['frame_length'], add_kwargs['hop_length'], 
                                            analysis_kwargs['frame_length'], analysis_kwargs['center'])
            if mode == "yin":
                f0 = align(librosa.yin(audio, sr=sr, **analysis_kwargs))
            elif mode == "pyin":
                f0, voiced_flag, voiced_prob = librosa.pyin(audio, sr=sr, **analysis_kwargs)
                f0 = np.stack([align(f0), align(voiced_flag)], axis=-2)
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
            hop_length = kwargs.get('hop_length') or cls.default_frame_length
            add_kwargs = {
                # pesto step sizes are in milliseconds
                "step_size": hop_length / sr * 1000,
                'model_name': kwargs.get('model', "mir-1k_g7"),
                'reduction': kwargs.get('reduction', 'alwa'),
                'num_chunks': kwargs.get('num_chunks', 8), 
                'convert_to_freq': kwargs.get('convert_to_freq', True)
            }
            batch_shape = audio.shape[:-1]
            audio_t = torch.from_numpy(np.asarray(audio)).float().reshape(-1, audio.shape[-1]) # type: ignore
            if device is not None: audio_t = audio_t.to(device)
            _, preds, confidence, _ = pesto.predict(audio_t, sr, **add_kwargs)
            # (f0, confidence) along the before-last axis, as pyin
            f0 = torch.stack([preds, confidence], dim=-2).cpu().numpy()
            f0 = np.stack([_reframe_pitch_out(f0[:, i], audio, frame_length=kwargs.get('frame_length') or cls.default_frame_length, 
                                              win_length=hop_length, hop_length=hop_length) for i in range(2)], axis=-2)
            f0 = f0.reshape(*batch_shape, 2, f0.shape[-1])
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
    
    def script(self): 
        frame_length = self.kwargs.get('frame_length', self.default_frame_length) or self.default_frame_length
        hop_length = self.kwargs.get('hop_length', self.kwargs.get('frame_length', self.default_frame_length)) or self.default_frame_length

        if hasattr(self, "fmin"): 
            fmin = self.fmin
        else:
            fmin = self.kwargs.get('fmin', 'C2')
            if isinstance(fmin, str): fmin = librosa.note_to_hz(fmin)
        if hasattr(self, "fmax"):
            fmax = self.fmax
        else:
            fmax = self.kwargs.get('fmax', 'C7')
            if isinstance(fmax, str): fmax = librosa.note_to_hz(fmax)

        if self.mode == "yin":
            return ScriptableF0_Yin(sr=self.sr,
                                    fmin=float(fmin),
                                    fmax=float(fmax),
                                    frame_length=int(frame_length),
                                    win_length=int(frame_length) // 2,
                                    hop_length=int(hop_length),
                                    center=False,
            )
        elif self.mode == "crepe": 
            return ScriptableF0_Crepe(
                self.sr, 
                hop_length = hop_length, 
                fmin = float(self.fmin), 
                fmax = float(self.fmax)
            )
        elif self.mode == "pesto": 
            return ScriptableF0_Pesto(
                sr = self.sr,
                hop_length = hop_length,
                fmin=float(self.fmin),
                fmax=float(self.fmax)
            )
        else:
            raise NotImplementedError('mode %s not available for scripting'%self.mode)

    def __call__(self, x):
        out = self.predict(x.detach().cpu().numpy(), self.mode, self.sr, device=self.device, **self.kwargs)
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
    

@gin.configurable(module="ad.features")
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
            # pitch classes are on the first row of (..., 2, frames) outputs (the second row is voicing)
            pitch = Counter(np.asarray(meta)[..., 0, :].flatten().tolist()).most_common(1)[0][0]
        else:
            pitch = Counter(meta.flatten().tolist()).most_common(1)[0][0]
        return Pitch.note_hash()[pitch]

    @staticmethod
    def predict(audio: np.ndarray, mode: str, sr: int, **add_kwargs):
        f0 = F0.predict(audio, mode, sr, **add_kwargs)
        note = apply_nested(f0_to_pitch, f0.tolist())
        return np.array(note)


    def __call__(self, x):
        out = self.predict(x.detach().cpu().numpy(), self.mode, self.sr, device=self.device, **self.kwargs)
        out = torch.from_numpy(out).to(x.device)
        return out


class ScriptableF0_Yin(torch.nn.Module):
    """
    TorchScript-compatible YIN fundamental frequency estimator.

    Mirrors librosa.yin (yin mode only): computes the windowed autocorrelation
    difference function, its cumulative mean normalized form (CMNDF), then
    picks the first trough below trough_threshold and refines it with
    parabolic interpolation.

    fmin / fmax are in Hz (use librosa.note_to_hz to convert note names).
    Defaults match the yin_config used by F0 / Pitch:
      fmin = 65.41 Hz (C2), fmax = 2093.0 Hz (C7).

    Returns f0 in Hz with shape (..., n_frames).
    Unvoiced frames (no trough found below threshold) are set to 0.
    """

    sr: int
    fmin: float
    fmax: float
    frame_length: int
    win_length: int
    hop_length: int
    trough_threshold: float
    center: bool

    def __init__(
        self,
        sr: int = 44100,
        fmin: float = 65.41,    # C2
        fmax: float = 2093.0,   # C7
        frame_length: int = 2048,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        trough_threshold: float = 0.1,
        center: bool = True,
    ):
        super().__init__()
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.frame_length = frame_length
        self.win_length = win_length if win_length is not None else frame_length // 2
        self.hop_length = hop_length if hop_length is not None else frame_length // 4
        self.trough_threshold = trough_threshold
        self.center = center

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (..., samples) float; returns (..., n_frames) f0 in Hz."""
        W = self.frame_length
        win = self.win_length
        H = self.hop_length
        sr = self.sr

        if self.center:
            audio = torch.nn.functional.pad(audio, (W // 2, W // 2))

        # (..., n_frames, W)
        frames = audio.unfold(-1, W, H)
        batch_shape = list(frames.shape[:-1])   # (..., n_frames)
        frames_flat = frames.reshape(-1, W)      # (N, W)
        N = frames_flat.shape[0]

        min_period = max(1, int(sr / self.fmax))
        max_period = max(min_period, min(int(sr / self.fmin), max(0, W - win - 1)))

        # --- Windowed autocorrelation via FFT cross-correlation ---
        # acf[tau] = sum_{j=0}^{win-1} x[j] * x[j+tau]
        n_fft = 2 * W
        X_win  = torch.fft.rfft(frames_flat[:, :win], n=n_fft, dim=-1)
        X_full = torch.fft.rfft(frames_flat,           n=n_fft, dim=-1)
        acf = torch.fft.irfft(X_win.conj() * X_full, n=n_fft, dim=-1)
        acf = acf[:, :max_period + 1]           # (N, max_period+1)

        # --- Rolling window energy: energy_at[tau] = sum_{j=tau}^{tau+win-1} x[j]^2 ---
        prefix = torch.cat([
            torch.zeros([N, 1], dtype=frames_flat.dtype, device=frames_flat.device),
            torch.cumsum(frames_flat ** 2, dim=-1),
        ], dim=-1)                               # (N, W+1); prefix[i] = sum_{j<i} x[j]^2
        energy_at = (prefix[:, win : win + max_period + 1]
                     - prefix[:, : max_period + 1])   # (N, max_period+1)
        energy_0 = energy_at[:, :1]              # (N, 1) reference window

        # --- Difference function d[tau] = energy_0 + energy_at[tau] - 2*acf[tau] ---
        d = energy_0 + energy_at - 2.0 * acf    # (N, max_period+1)
        d[:, 0] = 0.0

        # --- CMNDF: cmndf[tau] = tau * d[tau] / sum_{k=1}^{tau} d[k] ---
        d_cumsum = torch.cumsum(d[:, 1:], dim=-1)   # (N, max_period)
        tau_range = torch.arange(1, max_period + 1, dtype=d.dtype, device=d.device)
        cmndf = torch.cat([
            torch.ones([N, 1], dtype=d.dtype, device=d.device),
            d[:, 1:] * tau_range / (d_cumsum + 1e-10),
        ], dim=-1)                               # (N, max_period+1)

        # --- Peak picking: first local min below threshold in [min_period, max_period] ---
        win_size = max_period - min_period + 1
        cmndf_win = cmndf[:, min_period : max_period + 1]  # (N, win_size)

        if win_size >= 3:
            interior = cmndf_win[:, 1:-1]
            is_lm = ((interior < cmndf_win[:, :-2])
                     & (interior < cmndf_win[:, 2:])
                     & (interior < self.trough_threshold))
            local_min = torch.zeros([N, win_size], dtype=torch.bool, device=d.device)
            local_min[:, 1:-1] = is_lm
        else:
            local_min = cmndf_win < self.trough_threshold

        pos   = torch.arange(win_size, dtype=torch.long, device=d.device)
        large = torch.full([N, win_size], win_size, dtype=torch.long, device=d.device)
        masked    = torch.where(local_min, pos.unsqueeze(0).expand(N, win_size), large)
        first_idx = masked.min(dim=-1).values          # (N,)
        found     = first_idx < win_size               # (N,)

        global_argmin = cmndf_win.argmin(dim=-1)       # (N,)
        tau_in_win = torch.where(found, first_idx, global_argmin)
        actual_tau = tau_in_win + min_period           # (N,) lag index into cmndf

        # --- Parabolic interpolation ---
        tau_m = (actual_tau - 1).clamp(min=0)
        tau_p = (actual_tau + 1).clamp(max=max_period)

        a = cmndf.gather(1, tau_m.unsqueeze(1)).squeeze(1)
        b = cmndf.gather(1, actual_tau.unsqueeze(1)).squeeze(1)
        c = cmndf.gather(1, tau_p.unsqueeze(1)).squeeze(1)

        denom = a - 2.0 * b + c
        shift = torch.where(
            denom.abs() > 1e-6,
            0.5 * (a - c) / denom,
            torch.zeros_like(denom),
        )
        refined_tau = actual_tau.float() + shift       # (N,)

        # --- Lag → Hz, unvoiced → 0 ---
        f0 = float(sr) / refined_tau.clamp(min=1.0)
        f0 = f0.clamp(min=self.fmin, max=self.fmax)
        f0 = torch.where(found, f0, torch.zeros_like(f0))

        return f0.reshape(batch_shape)


class ScriptablePitch_Yin(torch.nn.Module):
    """
    TorchScript-compatible pitch-class estimator built on ScriptableF0_Yin.

    Converts f0 (Hz) to the pitch-class index used by Pitch.note_hash:
      -1: unvoiced, 0: A, 1: A♯, 2: B, 3: C, 4: C♯, 5: D,
       6: D♯, 7: E, 8: F, 9: F♯, 10: G, 11: G♯

    Returns an int64 tensor of shape (..., n_frames).
    """

    f0_module: ScriptableF0_Yin

    def __init__(self, f0_module: ScriptableF0_Yin):
        super().__init__()
        self.f0_module = f0_module

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (..., samples) float; returns (..., n_frames) pitch class int64."""
        f0 = self.f0_module(audio)                     # (..., n_frames) Hz
        voiced = f0 > 0.0

        # Hz → MIDI → C-based semitone class → A-based class
        midi    = 69.0 + 12.0 * torch.log2(f0.clamp(min=1e-6) / 440.0)
        c_based = midi.round().long() % 12             # C=0 … B=11
        a_based = (c_based - 9) % 12                  # A=0 … G♯=11
        return torch.where(voiced, a_based, torch.full_like(a_based, -1))


# ── CREPE ──────────────────────────────────────────────────────────────────

class ScriptableF0_Crepe(torch.nn.Module):
    """
    Clean nn.Module wrapper for CREPE pitch estimation.

    Replaces all numpy / librosa calls in F0(mode='crepe') with pure PyTorch:
      - resampling via torchaudio.transforms.Resample
      - framing and normalisation with unfold
      - weighted-argmax decoder (no viterbi / numpy)

    The CREPE backbone is stored as a traced module (torch.jit.trace).
    The wrapper itself can be torch.jit.trace-d but not torch.jit.script-ed
    (upstream Crepe.layer uses an un-annotated tuple as F.pad argument).

    fmin / fmax in Hz.  Default hop_length = 512 samples at the source sr.
    """

    hop_length_resampled: int
    fmin: float
    fmax: float
    pad: bool

    def __init__(
        self,
        sr: int = 44100,
        hop_length: int = 512,
        fmin: float = 50.0,
        fmax: float = 2006.0,
        model: str = 'full',
        pad: bool = True,
    ):
        super().__init__()
        try:
            import torchcrepe as _tc
        except ImportError:
            raise ImportError("torchcrepe is required: pip install torchcrepe")

        _CREPE_SR = 16000
        self.hop_length_resampled = int(round(hop_length * _CREPE_SR / sr))
        self.fmin = fmin
        self.fmax = fmax
        self.pad = pad

        self.resampler = torchaudio.transforms.Resample(sr, _CREPE_SR)

        # Trace the backbone so it survives torch.jit.trace on the wrapper
        _tc.load.model('cpu', model)
        backbone = _tc.infer.model.eval()
        self.backbone = torch.jit.trace(backbone, torch.zeros(1, _tc.core.WINDOW_SIZE))

    def _decode(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: (N, 360) → f0: (N,) Hz via weighted argmax."""
        bins = torch.arange(360, dtype=logits.dtype, device=logits.device)
        hz_all = 10.0 * (2.0 ** ((20.0 * bins + 1997.3794084376191) / 1200.0))

        # Suppress bins outside [fmin, fmax]
        valid = (hz_all >= self.fmin) & (hz_all <= self.fmax)
        logits = logits.masked_fill(~valid.unsqueeze(0), float('-inf'))

        # Weighted sum in a ±4-bin window around the peak
        peak = logits.argmax(dim=-1, keepdim=True)                    # (N, 1)
        pos  = torch.arange(360, device=logits.device).unsqueeze(0)   # (1, 360)
        logits = logits.masked_fill(~((pos >= peak - 4) & (pos <= peak + 4)), float('-inf'))

        probs     = torch.sigmoid(logits)
        cents     = 20.0 * bins + 1997.3794084376191
        f0_cents  = (cents.unsqueeze(0) * probs).sum(dim=-1) / probs.sum(dim=-1).clamp(min=1e-8)
        return 10.0 * (2.0 ** (f0_cents / 1200.0))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (..., samples) → (..., n_frames) f0 Hz."""
        W = 1024   # CREPE WINDOW_SIZE
        H = self.hop_length_resampled

        batch_shape = list(audio.shape[:-1])
        audio_2d = audio.reshape(-1, audio.shape[-1])   # (N, samples)
        audio_2d = self.resampler(audio_2d)

        if self.pad:
            audio_2d = torch.nn.functional.pad(audio_2d, (W // 2, W // 2))

        frames = audio_2d.unfold(-1, W, H)              # (N, n_frames, W)
        N, n_frames = frames.shape[0], frames.shape[1]
        frames_flat = frames.reshape(N * n_frames, W)

        # Mean-centre and unit-variance normalise (matches torchcrepe.preprocess)
        mu    = frames_flat.mean(dim=-1, keepdim=True)
        sigma = frames_flat.std(dim=-1, keepdim=True).clamp(min=1e-10)
        frames_flat = (frames_flat - mu) / sigma

        logits  = self.backbone(frames_flat)             # (N*n_frames, 360)
        f0_flat = self._decode(logits)                   # (N*n_frames,)
        return f0_flat.reshape(batch_shape + [n_frames])


class ScriptablePitch_Crepe(torch.nn.Module):
    """Pitch-class estimator built on ScriptableF0_Crepe. Same index as Pitch.note_hash."""

    f0_module: ScriptableF0_Crepe

    def __init__(self, f0_module: ScriptableF0_Crepe):
        super().__init__()
        self.f0_module = f0_module

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (..., samples) → (..., n_frames) pitch class int64."""
        f0      = self.f0_module(audio)
        voiced  = f0 > 0.0
        midi    = 69.0 + 12.0 * torch.log2(f0.clamp(min=1e-6) / 440.0)
        c_based = midi.round().long() % 12
        a_based = (c_based - 9) % 12
        return torch.where(voiced, a_based, torch.full_like(a_based, -1))


# ── PESTO ──────────────────────────────────────────────────────────────────

class ScriptableF0_Pesto(torch.nn.Module):
    """
    Clean nn.Module wrapper for PESTO pitch estimation.

    Compared to F0(mode='pesto'), removes the numpy-conversion round-trip and
    returns a properly shaped f0 tensor (the original code accidentally
    concatenated timestamps with f0 and confidence via torch.cat).

    The pesto model is loaded once in __init__ and stored as a submodule.
    Requires: pip install pesto-pitch

    fmin / fmax filtering is applied after prediction (PESTO returns Hz directly
    when convert_to_freq=True).
    """

    sr: int
    step_size: float
    reduction: str
    num_chunks: int
    fmin: float
    fmax: float

    def __init__(
        self,
        sr: int = 44100,
        hop_length: int = 512,
        model_name: str = 'mir-1k_g7',
        reduction: str = 'alwa',
        num_chunks: int = 8,
        fmin: float = 0.0,
        fmax: float = float('inf'),
    ):
        super().__init__()
        try:
            import pesto as _pesto
        except ImportError:
            raise ImportError("pesto is required: pip install pesto-pitch")

        self.model     = _pesto.load(model_name)
        self._model_name = model_name
        self.sr        = sr
        self.step_size = hop_length / sr          # seconds
        self.reduction = reduction
        self.num_chunks = num_chunks
        self.fmin      = fmin
        self.fmax      = fmax

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (..., samples) → (..., n_frames) f0 Hz (0 outside [fmin, fmax])."""
        import pesto as _pesto

        batch_shape = list(audio.shape[:-1])
        audio_2d = audio.reshape(-1, audio.shape[-1])   # (N, samples)

        # pesto.predict returns (timestamps, f0, confidence, activations)
        _, f0, _, _ = _pesto.predict(
            audio_2d, self.sr,
            step_size=self.step_size,
            model_name=self._model_name,
            reduction=self.reduction,
            num_chunks=self.num_chunks,
            convert_to_freq=True,
        )                                               # f0: (N, n_frames)

        # Suppress out-of-range bins
        f0 = torch.where(
            (f0 >= self.fmin) & (f0 <= self.fmax),
            f0,
            torch.zeros_like(f0),
        )
        return f0.reshape(batch_shape + [f0.shape[-1]])


class ScriptablePitch_Pesto(torch.nn.Module):
    """Pitch-class estimator built on ScriptableF0_Pesto. Same index as Pitch.note_hash."""

    f0_module: ScriptableF0_Pesto

    def __init__(self, f0_module: ScriptableF0_Pesto):
        super().__init__()
        self.f0_module = f0_module

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (..., samples) → (..., n_frames) pitch class int64."""
        f0      = self.f0_module(audio)
        voiced  = f0 > 0.0
        midi    = 69.0 + 12.0 * torch.log2(f0.clamp(min=1e-6) / 440.0)
        c_based = midi.round().long() % 12
        a_based = (c_based - 9) % 12
        return torch.where(voiced, a_based, torch.full_like(a_based, -1))
