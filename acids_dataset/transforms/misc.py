import torch
import pdb
import math
import random
import gin
import bisect

import torchaudio
from torchaudio.functional import lfilter

from .base import Transform
from typing import Tuple
import numpy as np
from .resample_poly import resample_poly
from .utils import random_phase_mangle, normalize_signal, derivate, integrate
from ..utils import pad, PadMode, mirror_pad, match_loudness
from scipy.stats import bernoulli
from scipy.signal import lfilter, butter, sosfilt


@gin.configurable(module="transforms")
class Resample(Transform):
    """
    Resample target signal to target sample rate.
    """
    takes_as_input = Transform.input_types.torch
    def __init__(self, sr: int | None = None, target_sr: int | None = None, **kwargs):
        super().__init__(sr=sr)
        assert self.sr is not None, "Resample needs the sr keyword"
        self.target_sr = target_sr or self.sr

    def apply(self, x):
        if (self.sr == self.target_sr): return x
        return torchaudio.functional.resample(x, self.sr, self.target_sr)

    @classmethod
    def test(cls):
        def test_fn(input, output):
            return True
        return [(Resample(44100, 22050), test_fn),
                (Resample(44100, 88200), test_fn),
                (Resample(44100, 32000), test_fn)]

@gin.configurable(module="transforms")
class PreEmphasis(Transform):
    takes_as_input = Transform.input_types.torch
    def __init__(self, coeff: float = 0.97, **kwargs): 
        super().__init__(**kwargs)
        self.coeff = coeff

    def apply(self, x):
        return torchaudio.functional.preemphasis(x, self.coeff)


@gin.configurable(module="transforms")
class Stretch(Transform):
    takes_as_input = Transform.input_types.torch
    def __init__(self, 
                 n_signal: int = None, 
                 pitch_range = None, 
                 max_factor: int = 20, 
                 p: float = 0.5, 
                 loudness_correction: bool = True,
                 **kwargs):
        super().__init__(p=p, **kwargs)
        assert self.sr is not None, "sr must be given for Stretch"
        self.n_signal = n_signal
        self.max_factor = max_factor
        self.pitch_range = pitch_range or [0.7, 1.3]
        self.factor_list, self.ratio_list = self._get_factors(self.max_factor, self.pitch_range)
        self.loudness_correction = loudness_correction

    def _get_factors(self, factor_limit, pitch_range):
        factor_list = []
        ratio_list = []
        for x in range(1, factor_limit):
            for y in range(1, factor_limit):
                if (x==y):
                    continue
                factor = x / y
                if factor <= pitch_range[1] and factor >= pitch_range[0]:
                    i = bisect.bisect_left(factor_list, factor)
                    factor_list.insert(i, factor)
                    ratio_list.insert(i, (x, y))
        return factor_list, ratio_list

    def apply_random(self, x):
        if random.random() < self.p:
            return self.__call__(x)
        else:
            return x

    def _pitch_signal(self, x, pitch):
        ratio_idx = bisect.bisect_left(self.factor_list, pitch)
        if ratio_idx == len(self.factor_list):
            ratio_idx -= 1
        up, down = self.ratio_list[ratio_idx]
        ndim = x.ndim
        if ndim == 1: x = x[None, None]
        if ndim == 2: x = x[None]
        x_pitched = resample_poly(x, up, down, axis=-1)
        if ndim == 1: x_pitched = x_pitched[0, 0]
        if ndim == 2: x_pitched = x_pitched[0]
        x_pitched = (x_pitched / x_pitched.max()) * x.max()
        if self.loudness_correction:
            x_pitched = match_loudness(x_pitched, x, self.sr)
        return x_pitched

    def apply(self, x):
        random_range = list(self.pitch_range)
        random_range[1] = min(random_range[1], x.shape[-1] / (self.n_signal or x.shape[-1]))
        random_pitch = random.random() * (random_range[1] - random_range[0]) + random_range[0]
        x_pitched = self._pitch_signal(x, random_pitch)
        return x_pitched

@gin.configurable(module="transforms")
class Log1p(Transform):
    def __init__(self, contrast=1.0, p = 1.0, **kwargs):
        super().__init__(p=p, **kwargs)
        self.contrast = contrast
    def apply(self, x):
        return torch.log(1 + torch.clamp(x * self.contrast, 1e-5))

@gin.configurable(module="transforms")
class Exp1m(Transform):
    def __init__(self, contrast=1.0, p = 1.0, **kwargs):
        super().__init__(p=p, **kwargs)
        self.contrast = contrast
    def apply(self, x):
        return torch.exp(x) / self.contrast - 1


@gin.configurable(module="transforms")
class Crop(Transform):
    """
    Randomly crops signal to fit n_signal samples
    """
    def __init__(self, n_signal: int | None = None, dim=-1, pad_if_shorter=None, pad_dir="right", **kwargs):
        super().__init__(**kwargs)
        self.n_signal = n_signal
        self.pad_if_shorter = pad_if_shorter
        self.dim = dim
        self.pad_dir = pad_dir

    def apply_random(self, x):
        if random.random() < self.p:
            return self.apply(x)
        else:
            return x

    def _pad(self, x):
        missing_points = self.n_signal - x.shape[self.dim]
        if self.pad_dir == "right": 
            pad_dim = (0, missing_points)
        elif self.pad_dir == "left": 
            pad_dim = (missing_points, 0)
        else:
            pad_dim = (math.floor(missing_points)/2, math.ceil(missing_points)/2)
        return torch.nn.functional.pad(x, pad_dim, mode=self.pad_if_shorter)

    def apply(self, x):
        if x.shape[self.dim] < self.n_signal: 
            if self.pad_if_shorter is None:
                raise ValueError("cannot crop %s samples on dim %d: got %d"%(self.n_signal, self.dim,  x.shape[self.dim]))
            else:
                x = self._pad(x)
        in_point = random.randint(0, x.shape[self.dim] - self.n_signal)
        idx = [slice(None)] * len(x.shape); idx[self.dim] = slice(in_point, in_point + self.n_signal)
        return x.__getitem__(tuple(idx))

    @classmethod
    def test(cls, dim=-1, crop_size=2048):
        def is_output_ok(input, output):
            return output.shape[dim] == crop_size
        return [(cls(n_signal=crop_size, dim=dim), is_output_ok)]
        
    def __call__(self, x): 
        if self.n_signal is None: 
            return x
        return super().__call__(x)
        

@gin.configurable(module="transforms")
class Dequantize(Transform):
    def __init__(self, bit_depth = 16, **kwargs):
        super().__init__(**kwargs)
        self.bit_depth = bit_depth

    def apply(self, x):
        if self.type_hash(x) == self.input_types.torch:
            x += torch.rand(*x.shape) / 2**self.bit_depth
        else:
            x += np.random.rand(*x.shape) / 2**self.bit_depth
        return x

@gin.configurable(module="transforms")
class Compress(Transform):
    takes_as_input = Transform.input_types.torch
    def __init__(self, threshold = -40, amp_range = [-60, 0], attack=0.1, release=0.1, prob=0.8, limit=True, sr=44100, **kwargs):
        super().__init__(sr=sr, **kwargs)
        assert prob >= 0. and prob <= 1., "prob must be between 0. and 1."
        self.amp_range = amp_range
        self.threshold = threshold
        self.attack = attack
        self.release = release
        self.prob = prob
        self.limit = limit
        self.sr = sr

    def apply(self, x: torch.Tensor):
        batch_size = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        amp_factor = torch.rand((1,)) * (self.amp_range[1] - self.amp_range[0]) + self.amp_range[0]
        x_aug = torchaudio.sox_effects.apply_effects_tensor(x,
                                                            self.sr,
                                                            [['compand', f'{self.attack},{self.release}', f'6:-80,{self.threshold},{float(amp_factor)}']]
                                                            )[0]
        if (self.limit) and (x_aug.abs().max() > 1): 
            x_aug = x_aug / x_aug.abs().max()
        return x_aug.reshape(batch_size + (x.shape[-1],))


@gin.configurable(module="transforms")
class Gain(Transform):
    takes_as_input = Transform.input_types.torch
    def __init__(self, gain_range: Tuple[int, int] | None = None, limit = True, **kwargs):
        super().__init__(**kwargs)
        self.gain_range = gain_range or [-12, 3]
        self.limit = limit

    def apply(self, x: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        gain_factor = torch.rand(1)[None, None][0] * (self.gain_range[1] - self.gain_range[0]) + self.gain_range[0]
        amp_factor = torch.pow(10, gain_factor / 20)
        x_amp = x * amp_factor
        if (self.limit) and (torch.abs(x_amp).max() > 1): 
            x_amp = x_amp / torch.abs(x_amp).max()
        return x_amp


@gin.configurable(module="transforms")
class Mute(Transform):
    def __init__(self, p: float = 0.1, noise_std: float = 1.e-5, **kwargs):
        super().__init__(p=p, **kwargs)
        self.noise_std = noise_std

    def apply(self, x):
        if self.type_hash(x) == self.input_types.torch:
            if self.noise_std > 0:
                return torch.randn_like(x) * self.noise_std
            else:
                return torch.zeros_like(x)
        else:
            if self.noise_std > 0:
                return np.random.random(size=x.shape) * self.noise_std
            else:
                return np.zeros_like(x)

@gin.configurable(module="transforms")
class PhaseMangle(Transform):
    takes_as_input = Transform.input_types.torch
    def __init__(self, 
                 p: float = 0.8,
                 min_f: float = 20, 
                 max_f: float = 2000, 
                 amplitude: float = .99, 
                 **kwargs):
        super().__init__(p=p, **kwargs)
        self.min_f = min_f
        self.max_f = max_f
        self.amplitude = amplitude

    def __repr__(self):
        return f"PhaseMangle(p={self.p}, min_f={self.min_f}, max_f={self.max_f}, amplitude={self.amplitude}, sr={self.sr})"

    def apply(self, x):
        return random_phase_mangle(x, self.min_f, self.max_f, self.amplitude, self.sr)


@gin.configurable(module="transforms")
class FrequencyMasking(Transform):
    takes_as_input = Transform.input_types.torch
    def __init__(self, p = 0.5, n_fft: int = 4096, max_size: int = 512, **kwargs):
        super().__init__(p=p, **kwargs)
        self.max_size = max_size
        self.n_fft = n_fft

    def apply(self, x: torch.Tensor):
        batch_size = x.shape[:-1]
        n_samples = x.shape[-1]
        x = x.reshape(-1, x.shape[-1])
        n_windows = (n_samples - self.n_fft) / math.floor(self.n_fft / 4)
        target_size = math.ceil(n_windows) * math.floor(self.n_fft / 4) + self.n_fft
        x = pad(x, target_size, PadMode.REFLECT)
        # pad with zeros
        spectrogram = x.stft(self.n_fft, return_complex = True)
        mask_size = random.randrange(1, self.max_size)
        freq_idx = random.randrange(0, spectrogram.shape[-2] - mask_size)
        spectrogram[..., freq_idx:freq_idx+mask_size, :] = 0
        x_inv = torch.istft(spectrogram, self.n_fft)[..., :n_samples]
        return x_inv.reshape(batch_size+(x_inv.shape[-1],))

    @classmethod
    def test(cls):
        def is_out_ok(input, output): 
            return input.shape[-1] == output.shape[-1]
        return [(cls(), is_out_ok)]
            
@gin.configurable(module="transforms")
class RandomEQ(Transform):
    """
    Random equalization. From Victor Shepardson's fork of RAVE (https://github.com/victor-shepardson/RAVE)

    Args:
        Transform (_type_): _description_
    """
    takes_as_input = Transform.input_types.numpy
    def __init__(self, p_lp=0.75, p_bp=0.5, n_bp=2, p_ls=0.5, **kwargs):
        """
        Random parametric EQ roughly simulating electric guitar 
        body+pickup resonances and tone control.
        Args:
            sr: audio sample rate
            p_lp: probability of applying lowpass filter
            p_bp: probability of applying each band filter
            n_pp: number of band filters
            p_ls: probability of applying low shelf filter
        """
        super().__init__(**kwargs)
        assert self.sr is not None, "RandomEQ needs a sample rate"
        self.p_lp = p_lp
        self.p_bp = p_bp
        self.n_bp = n_bp
        self.p_ls = p_ls 

    def apply(self, x):
        if bernoulli.rvs(self.p_lp):
        # if True:
            # low pass ~ 80-20k Hz
            f = (80 * 2 ** (8*random.random())) / (44100 / self.sr)
            sos = butter(1, f, 'lp', fs=self.sr, output='sos')
            x = sosfilt(sos, x)
        if bernoulli.rvs(self.p_ls):
            # low shelf ~ 40-640 Hz
            f = 40 * 2 ** (4*random.random()) / (44100 / self.sr)
            # gain is distributed as 1-sqrt(u)
            # median of about -11db, 95% about -32db
            w = random.random()**0.5
            sos = butter(1, f, 'lp', fs=self.sr, output='sos')
            x = x - w*sosfilt(sos, x)
        for _ in range(self.n_bp):
            if bernoulli.rvs(self.p_bp):
                # band ~ 160-5k Hz
                f = 160 * 2 ** (5*random.random()) / (44100 / self.sr)
                sos = butter(1, (f*2/3,f*3/2), 'bp', fs=self.sr, output='sos')
                # gain between 0 and 3
                # i.e. minimum -inf (notch), median 0db, max 9.5db
                w = random.random()**2 * 4 - 1
                x = x + w*sosfilt(sos, x)
        return x

    @classmethod
    def test(cls, n=100):
        def run_several_times(input, output):
            # try different settings
            for i in range(n):
                out = cls(sr=44100)(input)
            return True
        return [(cls(sr=44100), run_several_times)]


@gin.configurable(module="transforms")
class RandomDelay(Transform):
    takes_as_input = Transform.input_types.torch
    def __init__(self, max_delay:float=1024, **kwargs):
        """
        Random short comb-filtering delays. From Victor Shepardson's fork of RAVE : https://github.com/victor-shepardson/RAVE

        Args:
            p (float | None): transformation probability (default: None)
            max_delay: in samples. 
            signal length must be <= preprocessing length - max_delay
        """
        super().__init__(**kwargs)
        self.max_delay = max_delay

    def apply(self, x):
        d = random.random() * (self.max_delay-1)
        d_lo = int(d)+1
        d_hi = d_lo+1
        l = d - d_lo
        delayed = x[..., 1:-d_lo]*(1-l) + x[..., :-d_hi]*l
        mix = (random.random()*2-1)**3
        return mirror_pad(x[..., d_hi:] + delayed*mix, x.shape[-1])
@gin.configurable(module="transforms")
class RandomDistort(Transform):
    takes_as_input = Transform.input_types.numpy 
    def __init__(self, sr, max_drive=32, p_lp=0.75, p_bp=0.5, n_bp=2, p_ls=0.5, **kw):
        """Random distortion (EQ+gain+tanh). From Shepardson's fork of RAVE : https://github.com/victor-shepardson/RAVE"""
        super().__init__(**kw)
        self.eq = RandomEQ(sr=sr, p_lp=p_lp, p_bp=p_bp, n_bp=n_bp, p_ls=p_ls)
        self.max_drive = max_drive

    def apply(self, x):
        mix = random.random()**2
        x_eq = self.eq(x)
        # normalize to peak at 1 before distortion
        # (but max gain of 32 here)
        norm = min(1/np.max(np.abs(x_eq)), 32)
        # drive
        drive = 1/4 + random.random()**3 * (self.max_drive-1/4)
        # normalize back to original range and mix
        return np.tanh(x_eq*norm*drive)/norm * mix + x * (1-mix)


@gin.configurable(module="transforms")
class Normalize(Transform):
    takes_as_input = Transform.input_types.torch
    def apply(self, x):
        return normalize_signal(x)

@gin.configurable(module="transforms")
class Derivator(Transform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def apply(self, x):
        return derivate(x)

@gin.configurable(module="transforms")
class Integrator(Transform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.sr is not None, "sr is required for %s"%(type(self).__name__)
    def apply(self, x):
        return integrate(x, self.sr)


@gin.configurable(module="transforms")
class AddNoise(Transform):
    takes_as_input = Transform.input_types.torch
    def __init__(self, std=1.e-5, **kwargs):
        super().__init__(**kwargs)
        self.std = std
    def apply(self, x):
        return x + torch.randn_like(x) * 1.e-5
