import math
import torchaudio
import torch

from scipy.signal import firwin


_upfirdn_modes = [
    'constant', 'wrap', 'edge', 'smooth', 'symmetric', 'reflect',
    'antisymmetric', 'antireflect', 'line',
]

def _output_len(len_h, in_len, up, down):
    """The output length that results from a given input"""
    return (((in_len - 1) * up + len_h) - 1) // down + 1

def upfirdn(h, x, up=1, down=1):
    x_up = torch.nn.functional.interpolate(x, scale_factor=up, mode="linear")
    # fft convolution (equal to conv1d, i.e. correlation with h) : direct convolution of long upsampled signals with long
    # filters takes several GB on CPU
    x_filtered = torchaudio.functional.fftconvolve(x_up, h.flip(0).reshape((1,) * (x_up.ndim - 1) + (-1,)).to(x_up), mode="valid")
    x_down = torch.nn.functional.interpolate(x_filtered, scale_factor=1 / down, mode="linear")
    return x_down


def resample_poly(x, up, down, axis=0, window=('kaiser', 5.0),
                  padtype='constant', cval=None):
    
    if up != int(up):
        raise ValueError("up must be an integer")
    if down != int(down):
        raise ValueError("down must be an integer")
    up = int(up)
    down = int(down)
    if up < 1 or down < 1:
        raise ValueError('up and down must be >= 1')
    if cval is not None and padtype != 'constant':
        raise ValueError('cval has no effect when padtype is ', padtype)

    # Determine our up and down factors
    # Use a rational approximation to save computation time on really long
    # signals
    g_ = math.gcd(up, down)
    up //= g_
    down //= g_
    if up == down == 1:
        return x.clone()
    n_in = x.shape[axis]
    n_out = n_in * up
    n_out = n_out // down + bool(n_out % down)

    if isinstance(window, (list | torch.Tensor)):
        window = window.clone()  # use array to force a copy (we modify it)
        if window.ndim > 1:
            raise ValueError('window must be 1-D')
        half_len = (window.size - 1) // 2
        h = window
    else:
        # Design a linear-phase low-pass FIR filter
        max_rate = max(up, down)
        f_c = 1. / max_rate  # cutoff of FIR filter (rel. to Nyquist)
        half_len = 10 * max_rate  # reasonable cutoff for sinc-like function
        if x.dtype in [torch.float16, torch.float32, torch.float64]:
            h = torch.from_numpy(firwin(2 * half_len + 1, f_c, window=window)).to(x.dtype)
        else:
            h = torch.from_numpy(firwin(2 * half_len + 1, f_c, window=window))
    h *= up

    # Zero-pad our filter to put the output samples at the center
    n_pre_pad = (down - half_len % down)
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down
    # We should rarely need to do this given our filter lengths...
    while _output_len(len(h) + n_pre_pad + n_post_pad, n_in,
                      up, down) < n_out + n_pre_remove:
        n_post_pad += 1
    h = torch.cat((torch.zeros(n_pre_pad, dtype=h.dtype), h,
                   torch.zeros(n_post_pad, dtype=h.dtype)))
    n_pre_remove_end = n_pre_remove + n_out

    # Remove background depending on the padtype option
    funcs = {'mean': torch.mean, 'median': torch.median,
             'minimum': torch.amin, 'maximum': torch.amax}
    if padtype in funcs:
        background_values = funcs[padtype](x, axis=axis, keepdims=True)
    elif padtype in _upfirdn_modes:
        upfirdn_kwargs = {'mode': padtype}
        if padtype == 'constant':
            if cval is None:
                cval = 0
            upfirdn_kwargs['cval'] = cval
    else:
        raise ValueError(
            'padtype must be one of: maximum, mean, median, minimum, ' +
            ', '.join(_upfirdn_modes))

    if padtype in funcs:
        x = x - background_values

    # filter then remove excess
    y = upfirdn(h, x, up, down)
    keep = [slice(None), ]*x.ndim
    keep[axis] = slice(n_pre_remove, n_pre_remove_end)
    y_keep = y[tuple(keep)]

    # Add background back
    if padtype in funcs:
        y_keep += background_values

    return y_keep

