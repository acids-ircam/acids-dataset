# Features

Features (like audio descriptors) can be embedded in pre-processed datasets to make them accessible during training (see [Embedding features](preprocessing.md#embedding-features)), or computed dynamically in online mode by giving them to the dataset (`features=[...]`). Features can also be defined in [presets](presets.md), to be used in both modes. Feature configurations are contained in `acids_dataset/configs/features`, and can be customized (see [Customizing features](extending.md#customizing-features)).

Among all the available features, the `RegexpFeature` is a special feature that extracts metadata from file paths. By providing `meta_regexp` patterns, that can be understood as glob patterns with placeholders contained withing double curvy braces. For example, let's imagine (like Slakh) that your dataset is organised as folows :

```text
dataset/
  Track00001/
    mix.flac
    stems/
      S01.flac
      S02.flac
      ...
  Track00002/
    mix.flac
    stems/
      S01.flac
      S02.flac
```

you can extract respectively the track ids and the instrument ids into `id` and `inst` features with the pattern `"Track{{id}}/stems/S{{inst}}.flac"` (see [Embedding features](preprocessing.md#embedding-features) for the command line, or give `meta_regexp=[...]` to an online dataset).

## Audio descriptors

Each descriptor is a feature (computed in pre-processing or online, and in [batches](loaders.md#batch-features) on GPU). They are computed on the mix of channels, or on each channel with `keep_channels=True` (that adds a channel axis). Per-frame descriptors use `frame_length` / `hop_length` (non-overlapping frames by default) and give `(..., n_frames)` ; frequency descriptors are given in Hz, or normalized (ratio of the Nyquist frequency) or in midi with `unit`. Silent frames give 0.

| Family | Feature | Output | Description |
| ------ | ------- | ------ | ----------- |
| spectral | `SpectralCentroid` | per frame, `unit` | center of mass of the spectrum (brightness) |
| | `SpectralSpread` | per frame, Hz / normalized | bandwidth around the centroid |
| | `SpectralSkewness`, `SpectralKurtosis` | per frame | asymmetry and peakedness of the spectrum |
| | `SpectralRolloff` | per frame, `unit` | frequency below which `percentile` (0.85) of the energy lies |
| | `SpectralFlatness` | per frame, [0, 1] (or dB) | geometric / arithmetic mean of the spectrum (~0 for tones, ~0.56 for white noise) |
| | `SpectralCrest` | per frame | max / mean of the spectrum |
| | `SpectralFlux` | per frame | spectral change between consecutive frames |
| | `SpectralContrast` | `(n_bands, n_frames)`, dB | peak / valley difference in octave bands |
| representations | `STFT` | `(K, n_frames)` | short-time Fourier transform : complex, magnitude, power, dB, phase, real / imaginary parts, power density, instantaneous phase or frequency, see [Spectral representations and LPC](#spectral-representations-and-lpc) |
| | `DGT` | `(K, n_frames)` | discrete Gabor transform (STFT with gaussian windows, phase reconstruction with PGHI) |
| | `FFT` | `(K,)` | Fourier transform of the whole chunk (same modes, one-sided or full spectrum) |
| | `PQMF` | `(C * n_bands, T / n_bands)` | pseudo-QMF filter bank : critically sampled sub-bands (and their synthesis) |
| | `DCT` | `(n_coeffs, n_frames)` | discrete cosine transform of frames |
| | `Gammatone` | `(n_bands, n_frames)`, power or dB | energy in gammatone filters equally spaced on the ERB scale |
| | `Cepstrum` | `(n_coeffs, n_frames)` | real cepstrum : inverse Fourier transform of the log magnitude spectrum |
| | `LPC` | `(order, n_frames)` | linear prediction coefficients, reflection coefficients, line spectral frequencies or spectral envelope |
| temporal | `RMS` | per frame, dB (or linear) | root mean square energy |
| | `ZeroCrossingRate` | per frame, ratio or Hz | sign changes (in Hz : frequency of a sine with the same rate) |
| | `AttackTime` | per chunk, seconds | rise time of the envelope from 10% to 90% of its maximum |
| | `TemporalCentroid` | per chunk, seconds | center of mass of the envelope |
| | `CrestFactor` | per chunk, dB (or linear) | peak / RMS ratio |
| onsets | `OnsetStrength` | per frame | rectified flux of log-magnitude spectra |
| | `OnsetRate` | per chunk, onsets / s | peaks of onset strength (the first frame of a chunk cannot be an onset) |
| | `Tempo` | per chunk, BPM | autocorrelation of onset strength, with a prior around `prior_bpm` (120) |
| harmonic | `Chroma` | `(12, n_frames)` | energy of pitch classes (0 : C ... 11 : B) |
| | `Harmonicity` | per frame, [0, 1] | periodicity (normalized autocorrelation) |
| | `Inharmonicity` | per frame | mean relative deviation of partials from a harmonic series |
| spatial | `Width`, `SpatialCovariance`, `CrossSpectralMatrix`, `SpectroSpatialCovarianceVector` | see [Spatial features](#spatial-features) | stereo width, inter-channel covariances, cross spectral matrices, SSCV |
| | `InterChannelLevelDifference` | `(C - 1, n_frames)`, dB | level of each channel relative to the first one (per band with `n_bands`) |
| | `InterChannelTimeDifference` | `(C - 1, n_frames)`, samples or seconds | delay of each channel relative to the first one |
| | `InterChannelCoherence` | `(C - 1, n_frames)`, [0, 1] | maximum cross-correlation with the first channel |
| | `GeneralizedCrossCorrelation` | `(P, n_frames)`, samples or seconds (or `(P, n_lags, n_frames)`) | GCC-PHAT (and other weightings) delays between pairs of channels |
| | `Pan` | per frame, [-1, 1] | stereo position from the energies of the first two channels |
| quality | `Peak` | per chunk (per channel), linear (or dBFS) | maximum absolute amplitude |
| | `LoudnessMeter` | per chunk, `(n_values,)`, dB / dB(A) / LUFS | statistics of frame levels, see [Loudness](#loudness) |
| | `ClippingRatio` | per chunk | ratio of clipped samples |
| | `DCOffset` | per chunk (per channel) | mean value |
| | `DynamicRange` | per chunk, dB | difference between high and low percentiles of frame levels |
| | `SilenceRatio` | per chunk | ratio of silent frames |

```python
from acids_dataset.features import SpectralCentroid, Tempo, Chroma
dataset = AudioDataset("path/to/raw/audio", chunk_length=4., features=[SpectralCentroid(unit="normalized"), Tempo(), Chroma()])
```

Per-chunk descriptors can be used for [balancing](sampling.md#class-balancing) and [stratified partitions](partitions.md), as other continuous features.

## Loudness

`Loudness` gives the loudness in LUFS (per frame with `frame_length`, per channel with `keep_channels`). Values are floored to `min_loudness` (-70 LUFS by default, the absolute gate of loudness measurements), such that silence gives -70 instead of `-inf` (that would give NaN in losses) ; `min_loudness=None` keeps `-inf`.

`mode` sets the measure : `"lufs"` (default, ITU-R BS.1770 ; without `keep_channels`, combined loudness of channels), `"db"` for the RMS level in dBFS, or `"dba"` for the A-weighted RMS level in dB(A) (IEC 61672, 0 dB at 1 kHz). `db` and `dba` levels use the same frames and give the same shapes as LUFS ; without `keep_channels`, they are the levels of the mix of channels. Their default names are `loudness_db` and `loudness_dba`, such that several modes can be computed together, and they are also exported by `script()`. The same measures are available as functions in `acids_dataset.utils` : `level(x, sr, mode)`, `loudness`, `rms_level`.

```python
from acids_dataset.features import Loudness
dataset = AudioDataset("path/to/raw/audio", chunk_length=1., features=[Loudness(), Loudness(mode="dba", frame_length=2048)],
                       output_pattern="{waveform,loudness,loudness_dba}")
```

`LoudnessMeter` gives statistics of the levels of the frames of a chunk (`frame_length`, 2048, `hop_length`), concatenated in the last dimension in the order of `properties` :

| Level | Properties | Measure |
| ----- | ---------- | ------- |
| `db` | `min_db`, `max_db`, `db_quantiles` | RMS level, in dBFS |
| `dba` | `min_dba`, `max_dba`, `dba_quantiles` | A-weighted RMS level, in dB(A) (IEC 61672, 0 dB at 1 kHz) |
| `lufs` | `min_lufs`, `max_lufs`, `lufs_quantiles` | loudness, in LUFS (ITU-R BS.1770, as `Loudness`) |

```python
from acids_dataset.features import LoudnessMeter
meter = LoudnessMeter(properties=["min_db", "max_db", "lufs_quantiles"], n_quantiles=5, sr=44100)
meter(x)              # (..., 1 + 1 + 5)
meter.output_names    # ["min_db", "max_db", "lufs_q0", "lufs_q0.25", "lufs_q0.5", "lufs_q0.75", "lufs_q1"]
```

- `min_*` / `max_*` give 1 value each, `*_quantiles` give `n_quantiles` values (10 by default), at probabilities evenly spaced from 0 to 1 (the first one is the minimum, the last one the maximum). `output_names` labels the values of the output.
- Without `keep_channels`, `db` and `dba` levels are the ones of the mix of channels, and `lufs` the combined loudness of channels (BS.1770, up to 5 channels) ; with `keep_channels=True`, levels are measured on each channel, and outputs are `(..., C, n_values)`.
- Levels of silent frames are floored to `floor_db` (-120 by default) : in LUFS, frames below the absolute gate (-70 LUFS) are silent. Silent frames are included in statistics.
- Levels are measured on frames : they are not the integrated loudness of the chunk (given by `Loudness`).

## Spatial features

The mid / side transform and ambisonic encoding and decoding, that give signals instead of descriptors, are transforms : see [Spatial transforms](augmentations.md#spatial-transforms).

`Width`, `SpatialCovariance`, `CrossSpectralMatrix`, `SpectroSpatialCovarianceVector` and `GeneralizedCrossCorrelation` describe multichannel audio, per frame (`frame_length`, `hop_length`) :
- `Width` measures the stereo width from a mid / side analysis (mid = mean of channels, side = residual of each channel, i.e. (L + R) / 2 and ±(L - R) / 2 in stereo, generalized to any number of channels). `mode="width"` (default) gives side / (mid + side) energy in [0, 1] (0 for identical channels, 0.5 for uncorrelated channels, 1 for opposite channels), `mode="ratio"` the side to mid ratio in dB, and `mode="energy"` both energies in dB.
- `SpatialCovariance` gives the covariance matrices between channels, `(C, C, n_frames)`, normalized as correlations by default (`normalize=True`). With `n_bands`, matrices are computed on the spectrum and summed over `n_bands` linear frequency bands, with real and imaginary parts stacked : `(2, n_bands, C, C, n_frames)` (coherence matrices if normalized).
- `CrossSpectralMatrix` gives the cross spectral matrices (CSM) used in beamforming and acoustic imaging, per frequency bin : each frame (`frame_length`, 4096 by default) is estimated with Welch's method, averaging `X_i conj(X_j)` over hann-windowed segments of `n_fft` samples (1024) with 50% overlap, with a power spectral density scaling (`C[i, j]` equals `scipy.signal.csd(x_j, x_i)` on the frame). `mode="complex"` (default) stacks real and imaginary parts, `(2, K, C, C, n_frames)` ; `"magnitude"`, `"phase"` and `"coherence"` (magnitude squared coherence, in [0, 1]) give `(K, C, C, n_frames)`. Bins can be restricted with `fmin` / `fmax` (Hz) and averaged in `n_bands` linear bands, and their center frequencies are given by the `frequencies` attribute. Full-resolution matrices are large (`n_fft // 2 + 1` bins for each pair of channels) : restrict or band them before storing them in a pre-processed dataset.

```python
from acids_dataset.features import CrossSpectralMatrix
csm = CrossSpectralMatrix(n_fft=1024, fmin=100, fmax=8000, n_bands=32, mode="coherence", sr=44100)
dataset = AudioDataset("path/to/multichannel/audio", chunk_length=2., features=[csm])
```
- `SpectroSpatialCovarianceVector` is the Spectro-Spatial Covariance Vector (SSCV) of [Meng et al.](https://arxiv.org/abs/2411.03172), designed for first-order ambisonics recordings (and applicable to any number of channels `M`). Covariance matrices are computed in `n_bands` mel bands (52), on hann-windowed frames of 96 ms with 50% overlap by default, and optionally smoothed over frames with a one-pole filter (`alpha`, 0 by default : the paper learns it jointly with the model). Each matrix is turned into `M^2` real values : a real DFT of its diagonal, and the real and imaginary parts of its off-diagonal entries, normalized by the energy of the band, whose log is given first. Output : `(M^2, n_bands, n_frames)`, i.e. `(16, 52, n_frames)` for FOA. All values but the first one are invariant to the gain of the signal.

```python
from acids_dataset.features import SpectroSpatialCovarianceVector
dataset = AudioDataset("path/to/foa/recordings", sr=16000, chunk_length=4., features=[SpectroSpatialCovarianceVector(sr=16000)])
```

`GeneralizedCrossCorrelation` gives the generalized cross-correlation (GCC, [Knapp & Carter, 1976](https://doi.org/10.1109/TASSP.1976.1162830)) between pairs of channels, for time delay estimation and sound source localization. Its cross-spectrum is weighted with `weighting` : `"phat"` (phase transform, default ; `beta < 1` gives PHAT-β), `"scot"`, `"roth"`, `"ht"` (Hannan-Thomson / maximum likelihood) or `"none"` (normalized cross-correlation). Spectra are averaged over segments of `n_fft` samples in each frame (`n_fft = frame_length` by default) : `"scot"`, `"roth"` and `"ht"` differ from `"phat"` only with several segments per frame (`frame_length >> n_fft`). Pairs are each channel with the first one (`pairs="reference"`, `P = C - 1`) or all pairs `i < j` (`pairs="all"`, `P = C (C - 1) / 2`). `mode="delay"` (default) gives the lag of the GCC peak within `±max_delay` (1 ms by default, `None` for all lags), with sub-sample resolution (parabolic interpolation, and upsampling with `interp`), in samples or seconds (`unit`) : `(P, n_frames)`. `mode="correlation"` gives the GCC itself, `(P, n_lags, n_frames)`, with lags in the `lags` attribute. Positive delays : the second channel of the pair is delayed.

```python
from acids_dataset.features import GeneralizedCrossCorrelation
gcc = GeneralizedCrossCorrelation(frame_length=4096, n_fft=1024, weighting="phat", max_delay=5e-4, interp=4, pairs="all", unit="seconds", sr=48000)
dataset = AudioDataset("path/to/microphone/array", sr=48000, chunk_length=2., features=[gcc])
```

## Spectral representations and LPC

`STFT`, `DGT`, `DCT`, `Gammatone`, `Cepstrum` and `LPC` are computed on windowed frames (`frame_length`, `hop_length`, `window` : `"hann"` by default, `"hamming"` or `None`). Like other descriptors, they describe the mix of channels, or each channel with `keep_channels=True`, and give `(..., n, n_frames)`.

- `STFT` gives the short-time Fourier transform of frames of `frame_length` samples, zero-padded to `n_fft` (`frame_length` by default), every `hop_length` samples (`frame_length // 4` by default) ; `center=True` pads signals with `frame_length // 2` zeros on both sides. Its `mode` is one mode or a list of modes :
    - `"complex"`, `"magnitude"` (default), `"power"`, `"db"` (10 log10 of the power), `"phase"`, `"real"`, `"imaginary"` ;
    - `"power_density"` : power spectral density in V²/Hz, as `scipy.signal.spectrogram(scaling="density")` ;
    - `"instantaneous_phase"` : phase difference between consecutive frames, wrapped in [-π, π] (time derivative of the unwrapped phase, as the instantaneous frequency of GANSynth up to a factor π) ;
    - `"instantaneous_frequency"` : frequency (Hz) of each bin, from the phase difference between frames (phase vocoder) : a stationary sinusoid gives its exact frequency around its peak.

  Several modes are stacked on the channel axis, modes first : `(..., n_modes * C, K, n_frames)`, or `(..., n_modes, K, n_frames)` for the mix of channels. Outputs are real, unless `"complex"` is given (other modes are then cast to complex values). Frequencies of bins are given by the `frequencies` attribute.

  `STFT` is computed with [acids_transforms](https://github.com/acids-ircam/acids_transforms), and can be inverted, scripted and streamed for real-time use :
    - `feature.invert(features, inversion_mode=None, length=None)` gives back audio (`(..., T)` for the mix of channels, `(..., C, T)` with `keep_channels`, cropped or zero-padded to `length`) : exactly from `"complex"`, (`"real"`, `"imaginary"`) or a magnitude (`"magnitude"`, `"power"`, `"db"`, `"power_density"`) with `"phase"` ; approximately with instantaneous phases or frequencies ; and from magnitudes only with `inversion_mode` (`"griffin_lim"` by default, `"random"`, `"keep_input"` or `"sinebank"`).
    - `feature.script()` gives a module for `torch.jit.script`, with `forward` (the feature) and `invert`.
    - `feature.streaming()` gives a scriptable module computing the feature on chunks of audio `(..., C, T)` (`T` a multiple of `hop_length`) : each chunk gives its `T // hop_length` frames, that are the ones of the feature with `center=False` (after `frame_length // hop_length - 1` frames containing the initial zeros of the stream). `invert` gives back chunks of audio, delayed by `latency = frame_length - hop_length` samples ; call `reset()` between streams.

  ```python
  stft = STFT(frame_length=1024, hop_length=256, mode=["magnitude", "phase"], sr=44100)
  audio = stft.invert(stft(x))                                  # exact reconstruction
  stream = torch.jit.script(stft.streaming())
  for chunk in chunks:                                          # chunks of n * 256 samples
      frames = stream(chunk)                                    # (2, 513, n)
      out = stream.invert(frames)                               # chunk of audio, delayed by stream.latency samples
  ```
- `DGT` gives the discrete Gabor transform of [acids_transforms](https://github.com/acids-ircam/acids_transforms) : an STFT with a gaussian window of `n_fft` samples (1024), every `hop_length` samples (`n_fft // 4` by default), on frames centered on `t * hop_length` (signals are padded by reflection), with the modes `"complex"`, `"magnitude"` (default), `"power"`, `"db"`, `"phase"`, `"real"` and `"imaginary"`, stacked as in `STFT`. Gaussian windows suit phase reconstruction from magnitudes : `invert` uses PGHI (phase gradient heuristic integration) by default (see [Inverting features](#inverting-features)).
- `FFT` gives the Fourier transform of the whole chunk (optionally windowed, and cropped or zero-padded to `n_fft`), with the modes of `STFT` but instantaneous ones : `(..., K)`, with `K = n_fft // 2 + 1` (or the full spectrum of `n_fft` frequencies with `onesided=False`). `"power_density"` is the periodogram of the chunk (`scipy.signal.periodogram(scaling="density")`), and frequencies are given by `frequencies(n_samples)`. `invert` gives the chunk back from modes with phase.
- `PQMF` is a pseudo-QMF filter bank (as in multi-band vocoders and RAVE) : it splits `(..., C, T)` into `n_bands` (16) critically sampled sub-bands, `(..., C * n_bands, T / n_bands)` (bands of the first channel first ; signals are zero-padded to a multiple of `n_bands`), and `inverse=True` recombines sub-bands into signals. Filters are cosine modulations of a kaiser-windowed prototype (`attenuation`, 100 dB), optimized to cancel aliasing between bands : analysis followed by synthesis reconstructs signals with an error of about -60 dB, without delay (`invert` gives signals back from sub-bands).
- `DCT` gives the discrete cosine transform (type II, orthonormal) of each frame : the `n_coeffs` first coefficients of `scipy.fft.dct(frame, norm="ortho")` (all of them by default). `invert` gives audio back by inverse DCT and overlap-add (see [Inverting features](#inverting-features)).
- `Gammatone` gives the energy of frames in `n_bands` (64) gammatone filters, with center frequencies equally spaced on the ERB-rate scale between `fmin` (50 Hz) and `fmax` (`sr / 2`), given by its `center_frequencies` attribute. Filters are applied on the power spectrum of frames, with the magnitude response of gammatone filters of order 4 and bandwidth 1.019 ERB (Patterson et al.) : like `Mel`, it is an auditory spectrogram, with a finer resolution in low frequencies. `db=True` gives energies in dB. Its name is `gammatone_<n_bands>`.
- `Cepstrum` gives the real cepstrum of frames, `irfft(log |rfft(frame)|)`, with `n_coeffs` coefficients (`frame_length // 2 + 1` by default). Coefficient `q` is the quefrency `q / sr` (`quefrencies` attribute) : low coefficients describe the spectral envelope, and periodic signals of fundamental frequency `f0` give a peak at `sr / f0`.
- `LPC` computes linear predictive coding of order `order` (16), with the autocorrelation method and the Levinson-Durbin recursion. The prediction polynomial is `A(z) = 1 + a_1 z^-1 + ... + a_p z^-p` (`x[n] ≈ -Σ a_k x[n - k]`), and `mode` gives :
    - `"lpc"` (default) : the coefficients `a_1 ... a_p`, `(order, n_frames)` ;
    - `"reflection"` : the reflection (PARCOR) coefficients, in ]-1, 1[ ;
    - `"lsf"` : the line spectral frequencies, in radians (increasing, in ]0, π[), robust to quantization and interpolation ;
    - `"envelope"` : the LPC spectral envelope `gain² / |A(e^jω)|²` in dB, on the `frame_length // 2 + 1` bins of the power spectrum.

  `gain=True` prepends the gain of each frame (square root of the prediction error energy) to `"lpc"`, `"reflection"` and `"lsf"` outputs, and `pre_emphasis` (e.g. 0.97 for speech) filters signals with `x[n] - pre_emphasis x[n - 1]` before the analysis. Silent frames give `A(z) = 1`. Default names are `lpc` and `lpc_<mode>`.

```python
from acids_dataset.features import STFT, PQMF, Gammatone, Cepstrum, LPC
features = [STFT(frame_length=1024, mode=["db", "instantaneous_frequency"]), PQMF(n_bands=16),
            Gammatone(n_bands=64, db=True), Cepstrum(n_coeffs=40), LPC(order=24, mode="lsf", gain=True)]
dataset = AudioDataset("path/to/raw/audio", chunk_length=2., features=features)
```

## Inverting features

Invertible features (`feature.invertible`) give audio back from their outputs with `feature.invert(features, ...)` ; other features raise a `NotImplementedError`. The spatial transforms `MidSide`, `AmbisonicEncoder` and `AmbisonicDecoder` are also invertible (see [Spatial transforms](augmentations.md#spatial-transforms)). Inversions keep batch dimensions and devices, and `length` crops or zero-pads outputs to the length of the original signal (frame-based features lose the samples after their last frame).

| feature | `invert` gives | exact inversion |
| ------- | -------------- | --------------- |
| `STFT` | audio, `(..., T)` (or `(..., C, T)` with `keep_channels`) | from `"complex"`, (`"real"`, `"imaginary"`), or a magnitude with `"phase"` ; approximate with instantaneous phases or frequencies, and from magnitudes only with `inversion_mode` (`"griffin_lim"`, `"random"`, `"keep_input"`, `"sinebank"`) |
| `DGT` | audio | from `"complex"`, (`"real"`, `"imaginary"`), or a magnitude with `"phase"` ; from magnitudes only with `inversion_mode` (`"pghi"` by default, `"griffin_lim"`, `"random"`, `"keep_input"`, `"sinebank"`). Signals have `(n_frames - 1) * hop_length` samples |
| `FFT` | audio | from modes with phase (as `STFT`), with `window=None` or `"hamming"` (hann windows vanish on the first sample). Samples cropped by `n_fft` are lost ; give `length` for signals of odd length |
| `DCT` | audio, by inverse DCT and weighted overlap-add | with all coefficients (`n_coeffs=None`), and `window=None` or overlapping windows (e.g. `hop_length = frame_length // 2`) ; with hann windows, the `hop_length` first and last samples are inaccurate. Fewer coefficients low-pass frames |
| `PQMF` | signals from sub-bands (sub-bands from signals with `inverse=True`) | near perfect reconstruction (about -60 dB) |

```python
from acids_dataset.features import DGT, DCT, FFT

x = torch.randn(2, 44100)
dgt = DGT(n_fft=1024, hop_length=256, mode=["magnitude", "phase"], keep_channels=True)
assert torch.allclose(dgt.invert(dgt(x), length=44100)[..., :44032], x[..., :44032], atol=1e-4)   # exact
magnitude = DGT(n_fft=1024, hop_length=256)
audio = magnitude.invert(magnitude(x), length=44100)                       # phase reconstruction (PGHI)
dct = DCT(frame_length=1024, window=None, keep_channels=True)
assert torch.allclose(dct.invert(dct(x), length=44100)[..., :44032], x[..., :44032], atol=1e-4)
fft = FFT(mode="complex", keep_channels=True)
assert torch.allclose(fft.invert(fft(x), length=44100), x, atol=1e-4)
```

## Neural embeddings in online mode

`ModuleEmbedding` features compute embeddings with a PyTorch or TorchScript model. In online mode, they are computed in data loader workers :
- TorchScript models given with `module_path` are reloaded in each worker ; other modules must be given with `retain_module=True` to be sent to workers.
- Features are computed on CPU, unless a device is given : `device="cuda"`, `"mps"`, or `"auto"` for the default accelerated device (CUDA if available, unless `USE_CUDA=0` ; MPS if available and `USE_MPS=1` ; CPU otherwise). The device can be given to features (`ModuleEmbedding(..., device="cuda")`), to `preprocess_dataset` / `--device`, or to presets (`DEVICE`).
- Embeddings of `feature(x)` and `feature.batch_call(x)` are given on the device of `x`.
- `dataset.to(device)` (or `feature.to(device)`) moves features to a device. If features run on an accelerator, `get_loader` starts workers with `spawn`, as CUDA cannot be used in forked processes ; each worker then holds its own copy of the model. For heavy models, pre-processing embeddings (see [Computing embedding from a TorchScript model](preprocessing.md#computing-embedding-from-a-torchscript-model)) or computing them on batches (see [Batch features](loaders.md#batch-features)) is usually more efficient.

## Pitch features

`yin` and `pyin` pitch estimations need at least one period of the lowest frequency `fmin` in their analysis frames. If `frame_length` is too short for `fmin` (for example, `frame_length=512` with `fmin="C2"` at 44.1kHz), frames are analyzed with a larger window (two periods of `fmin`), while keeping the same `hop_length` : the number of output frames is unchanged. Inputs shorter than `frame_length` are zero-padded to a single frame (zero padding biases pitch estimation much less than reflection or repetition, that create phase discontinuities).
