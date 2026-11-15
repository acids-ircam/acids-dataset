from random import random
import gin.torch
import math
import numpy as np
import scipy
import torch, torchaudio


def random_angle(min_f=20, max_f=8000, sr=24000):
    min_f = math.log(min_f)
    max_f = math.log(max_f)
    rand = math.exp(random() * (max_f - min_f) + min_f)
    rand = 2 * torch.pi * rand / sr
    return rand

def pole_to_z_filter(omega, amplitude=.9):
    z0 = amplitude * torch.exp(1j * torch.tensor(omega))
    a = torch.Tensor([1, -2 * torch.real(z0), abs(z0)**2])
    b = torch.Tensor([abs(z0)**2, -2 * torch.real(z0), 1])
    return b, a

def random_phase_mangle(x, min_f, max_f, amp, sr, clamp=False):
    angle = random_angle(min_f, max_f, sr)
    b, a = pole_to_z_filter(angle, amp)
    return torchaudio.functional.lfilter(x, a, b, clamp=clamp)

def derivate(x, clamp=False):
    if torch.is_tensor(x):
        derivator = (torch.Tensor([.5, -.5]), torch.Tensor([1., 0.]))
        return torchaudio.functional.lfilter(x, derivator[1], derivator[0], clamp=clamp)
    else:
        derivator = ([.5, -.5], [1.])
        return scipy.signal.lfilter(*derivator, x)


def integrate(x, sr, clamp=False):
    if torch.is_tensor(x):
        alpha = 1 / (1 + 1 / sr * 2 * torch.pi * 10)
        # poles are close to the unit circle : computed in double precision
        b = torch.tensor([alpha**2, -alpha**2, 0], dtype=torch.float64)
        a = torch.tensor([1, -2 * alpha, alpha**2], dtype=torch.float64)
        return torchaudio.functional.lfilter(x.double(), a, b, clamp=clamp).to(x.dtype)
    else:
        alpha = 1 / (1 + 1 / sr * 2 * np.pi * 10)
        integrator = ([alpha**2, -alpha**2], [1, -2 * alpha, alpha**2])
        return scipy.signal.lfilter(*integrator, x)


def normalize_signal(x, max_gain_db: int = 30):
    peak = x.abs().amax()
    if peak == 0: return x

    log_peak = 20 * torch.log10(peak)
    log_gain = min(max_gain_db, -log_peak)
    gain = 10**(log_gain / 20)
    return x * gain
