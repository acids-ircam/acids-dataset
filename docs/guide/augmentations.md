# Augmentations

Data augmentations can work with both `np.ndarray` and `torch.Tensor`, and may be composed through the `transforms.Compose` that override usual list methods (`append` , `extend`, etc...). `acids-dataset` implements its own augmentations, and has a backend for the [audiomentations](https://github.com/iver56/audiomentations) package.

Stochastic options of the base class are very powerful to develop versatile augmentation pipelines. For example :

```python
from acids_dataset import transforms as atf

pipeline = atf.Compose([
  atf.Stretch(131072, p=0.2), # Stretch is always non-batchwise, as output lengths depend on the random factor
  atf.Gain(p=0.8), 
  atf.Compress(p=0.5), 
  atf.Mute(p=0.1), 
])
```

performs random resampling (speed and pitch change) with probability 20%, random gain with probability 80%, random compression with probability 50%, and mutes random batches with probability 10% (very useful to force models to learn silence). 

## Features and augmentations

By default, features computed by a dataset describe the **augmented** audio : they are computed on the waveform output after its transforms, item by item in workers as well as on batches (see [batch features](loaders.md#batch-features)). With `features_after_transforms=False`, features are computed on the loaded audio, **before** transforms, and describe the original audio :

```python
from acids_dataset import transforms as atf
from acids_dataset.features import Loudness

dataset = AudioDataset("path/to/raw/audio", chunk_length=131072, features=[Loudness()],
                       output_pattern="{waveform,loudness}", transforms={"waveform": atf.Gain(p=1.)})
dataset[0]["loudness"]                                                        # loudness of the audio with gain

dataset = AudioDataset("path/to/raw/audio", chunk_length=131072, features=[Loudness()], output_pattern="{waveform,loudness}",
                       transforms={"waveform": atf.Gain(p=1.)}, features_after_transforms=False)
dataset[0]["loudness"]                                                        # loudness of the original audio

dataset.features_after_transforms = True                                      # set the dataset
before = dataset.configure(features_after_transforms=False)                   # copy of the dataset
loader = dataset.get_loader(batch_size=16, features_after_transforms=False)   # for a single loader
```

- With several waveform entries (e.g. `{waveform->clean,waveform->x}` with `transforms={"x": ...}`), features are computed on the first waveform with a transform (here `x`) ; with `features_after_transforms=False`, on the loaded audio (here `clean`). Other waveform entries are given with their own transforms only.
- Transforms of features themselves (e.g. `transforms={"loudness": normalize}`) are always applied after their computation.
- Features of pre-processed datasets are read from the database : they describe the original audio, whatever `features_after_transforms`. Only features computed on batches (see [batch features](loaders.md#batch-features)) follow it.
- `CombinedAudioDataset(..., features_after_transforms=False)` sets every combined dataset (datasets given as objects are copied, and are not modified).

## Loudness range and peak normalization

`CompressRange` compresses (or expands) the loudness range of the input to `target_range` (in LUFS, or in the unit of `mode`) : the loudness of frames of `frame_duration` seconds (0.1) is mapped linearly from the input range to the target range, and the resulting gains are interpolated between frames. `NormRange` scales the input such that its peak equals `max_value`.

```python
from acids_dataset import transforms as atf

atf.CompressRange(target_range=(-30, -14), sr=44100)                      # input range measured on each input
atf.CompressRange(target_range=(-30, -14), input_range=(-60, 0), sr=44100)  # fixed input range
atf.CompressRange(target_range=(-40, -20), mode="dba", sr=44100)            # A-weighted RMS levels, in dB(A)
atf.NormRange(max_value=0.9)                                               # peak of every input set to 0.9
atf.NormRange(max_value=0.9, boost=False)                                  # only inputs above 0.9 are divided
```

- `mode` sets the loudness measure, as in `Loudness` : `"lufs"` (default), `"db"` (RMS level in dBFS) or `"dba"` (A-weighted RMS level in dB(A)). `target_range`, `input_range` and `silence_db` are given in its unit ; `db` and `dba` levels are the ones of the mix of channels.
- Without `input_range`, the input range is measured on each input (each item of a batch) : minimum and maximum loudness of its frames louder than `silence_db` (-70). Frames below the input range (resp. above) get the gain of its lower (resp. upper) bound, such that silence is not boosted more than quiet parts ; silent inputs are unchanged.
- Gains are the same for every channel (the stereo image is kept). As they are interpolated between frames, loudness only reaches the target range on steady parts (frames next to abrupt changes may differ by 1-2 dB). With `limit=True` (default), outputs whose peak exceeds 1 are scaled down to a peak of 1.
- `NormRange` measures the peak over channels, for each item of a batch : louder inputs are divided, and quieter ones multiplied (unless `boost=False`). Silent inputs are unchanged.

## Spatial transforms

`MidSide`, `AmbisonicEncoder` and `AmbisonicDecoder` (`acids_dataset.transforms`) change the channels of multichannel signals, `(..., C, T)` : they are deterministic (`p` is ignored), and can be used as dataset transforms, in `sample_features`, or called on tensors and arrays. They are invertible (`invertible = True`) : `invert` gives inputs back, as for [invertible features](features.md#inverting-features). Their default names are `midside` (`leftright` with `inverse=True`), `ambisonics` and `ambisonics_decoded`.

```python
from acids_dataset import transforms as atf
dataset = AudioDataset("path/to/stereo/audio", channels=2, chunk_length=131072, transforms=atf.MidSide())   # mid / side waveforms
```

`MidSide` is the mid / side transform of stereo signals : `(..., 2, T)` gives `(..., 2, T)`, mid first, with M = (L + R) / 2 and S = (L - R) / 2 (`normalization="average"`, default), or M = (L + R) / √2 and S = (L - R) / √2 (`normalization="orthonormal"`, energy preserving). `inverse=True` decodes mid / side signals into left / right signals (default name : `leftright`), and `invert` gives inputs back. Inputs that do not have 2 channels raise a `ValueError`.

### Ambisonics

`AmbisonicEncoder` and `AmbisonicDecoder` convert multichannel signals to and from ambisonics. Ambisonic signals follow the AmbiX convention : ACN channel ordering, real spherical harmonics without Condon-Shortley phase, and SN3D normalization (`normalization="n3d"` is also available). Directions are given as `(azimuth, elevation)` in degrees, azimuth counterclockwise from the front (90 : left) and elevation upwards.

Layouts are names of `acids_dataset.transforms.ambisonics.LAYOUTS` (`"mono"`, `"stereo"`, `"quad"`, `"5.0"`, `"7.0"`, `"octagon"`, `"tetrahedron"`, `"cube"`), or lists of directions.

- `AmbisonicEncoder(order, layout)` encodes `(..., C, T)` into `(..., (N + 1)², T)` : each input channel is a source in the direction given by `layout`. With `layout=None`, the layout is chosen from the number of channels : mono, stereo, or `C` directions equally spaced on the horizontal plane. `order` is `"adaptive"` (default) : the order is inferred from the number of channels, as the lowest order whose encoding can be inverted for the layout (mono gives order 0, stereo and tetrahedra order 1, quad and 5.0 order 2, 7.0 order 3), such that decoding on the same layout gives the input channels back. A fixed order (an integer) raises a `ValueError` for inputs that cannot be encoded invertibly at this order (e.g. 7.0 signals at order 1).
- `AmbisonicDecoder(layout, order)` decodes `(..., K, T)` to the `L` loudspeakers of `layout`, `(..., L, T)`. With `order="adaptive"` (default), the order is inferred from the number of input channels (`K = (N + 1)²`, otherwise a `ValueError` is raised) ; a fixed order `N` raises a `ValueError` for inputs that do not have `(N + 1)²` channels. With `layout=None`, signals are decoded to `(N + 1)²` virtual loudspeakers spread regularly on the sphere. `method="mode_matching"` (default) inverts the encoding of the layout (pseudo-inverse), while `method="sampling"` projects signals on the harmonics of each loudspeaker (for regular layouts).

```python
from acids_dataset.transforms import AmbisonicEncoder, AmbisonicDecoder
encoder = AmbisonicEncoder(layout="5.0")                  # 5 channels -> second order ambisonics (9 channels)
decoder = AmbisonicDecoder(layout="5.0")
dataset = AudioDataset("path/to/5.0/audio", channels=5, chunk_length=2., transforms=encoder)   # ambisonic waveforms
x = torch.randn(5, 44100)
assert torch.allclose(decoder(encoder(x)), x, atol=1e-4)  # exact reconstruction
```

The spherical harmonics of any directions are given by `acids_dataset.transforms.ambisonics.spherical_harmonics(order, azimuth, elevation)`, as an array `(n_directions, (order + 1)²)`. Encoders and decoders can also be inverted (as invertible features, see [Inverting features](features.md#inverting-features)) : they are each other's inverse. `encoder.inverse(n_channels)` gives the `AmbisonicDecoder` (mode matching) whose loudspeakers are the directions of the input channels, and `decoder.inverse(order)` the `AmbisonicEncoder` whose sources are the loudspeakers ; `invert` uses them.

```python
encoder = AmbisonicEncoder(layout="5.0")
decoder = encoder.inverse(5)                                  # AmbisonicDecoder(layout=<5.0 directions>, order=2, method="mode_matching")
assert torch.allclose(encoder.invert(encoder(x)), x, atol=1e-4)
encoder = AmbisonicDecoder(layout="cube").inverse(1)          # AmbisonicEncoder(order=1, layout=<cube directions>, strict=False)
```

Decoders often have more loudspeakers than ambisonic channels : `AmbisonicEncoder(strict=False)` then encodes channels that cannot be encoded invertibly at a fixed order (sources are mixed in ambisonic signals), instead of raising a `ValueError`.

For a full table of available transfroms, see at the [end](#full-augmentation-table) of `README.md`. 

## Full augmentation table

| Name                  | Config name                 | Description                                                                 |
| --------------------- | --------------------------- | --------------------------------------------------------------------------- |
| AddColorNoise         | `addcolornoise.gin`         | Add colored noise to audio signal                                           |
| AddGaussianNoise      | `addgaussiannoise.gin`      | Add gaussian noise to audio signal                                          |
| AddNoise              | `addnoise.gin`              | Add uniform noise to audio signal                                           |
| AddShortNoises        | `addshortnoises.gin`        | Add short bursts of noise to the signal                                     |
| AirAbsorption         | `airabsorption.gin`         | Add air absorption.                                                         |
| Aliasing              | `aliasing.gin`              | Aliases audio signal.                                                       |
| AmbisonicDecoder      | `ambisonicdecoder.gin`      | Decodes ambisonic signals to a loudspeaker layout.                          |
| AmbisonicEncoder      | `ambisonicencoder.gin`      | Encodes multichannel signals into ambisonics.                               |
| ApplyImpulseResponse  | `applyimpulseresponse.gin`  | Applies impulse response (needs audio files as first argument).             |
| BandPassFilter        | `bandpassfilter.gin`        | Random band-pass filter.                                                    |
| BasicPitch            | `basicpitch.gin`            | Pitch extraction based on BasicPitch                                        |
| Bitcrush              | `bitcrush.gin`              | Adds random bit-crushing (bit & sr reduction)                               |
| Clip                  | `clip.gin`                  | Clips incomping signal.                                                     |
| ClippingDistortion    | `clippingdistortion.gin`    | Adds clipping distortion.                                                   |
| Compress              | `compress.gin`              | Compression with random parameters.                                         |
| CompressRange         | `compressrange.gin`         | Compresses the loudness range of the signal to a target range (LUFS).       |
| Crop                  | `crop.gin`                  | Random cropping to fixed chunk size (shorter signals are padded).           |
| Dequantize            | `dequantize.gin`            | Dequantize incoming signal.                                                 |
| Derivator             | `derivator.gin`             | Derivates incoming signal.                                                  |
| FrequencyMasking      | `frequencymasking.gin`      | Randomly masks frequency bands of input.                                    |
| Gain                  | `gain.gin`                  | Applies random gain to the input.                                           |
| GainTransition        | `gaintransition.gin`        | Applies random gain ramps to the signal.                                    |
| HighPassFilter        | `highpassfilter.gin`        | Applies random high-pass filtering.                                         |
| HighShelfFilter       | `highshelffilter.gin`       | Applies random high-shelf filtering.                                        |
| Integrator            | `integrator.gin`            | Integrates incoming signal.                                                 |
| Lambda                | `lambda.gin`                | Performs a lambda function to the signal (needes lambda as first argument.) |
| LoudnessNormalization | `loudnessnormalization.gin` | Normalizes loundess according to IFRS standard.                             |
| LowPassFilter         | `lowpassfilter.gin`         | Applies random low-pass filtering.                                          |
| LowShelfFilter        | `lowshelffilter.gin`        | Applies random low-shelf filtering.                                         |
| MidSide               | `midside.gin`               | Mid / side transform of stereo signals (and its inverse).                   |
| Mp3Compression        | `mp3compression.gin`        | Applies random mp3 compression.                                             |
| Mute                  | `mute.gin`                  | Mutes incoming incoming signal.                                             |
| Normalize             | `normalize.gin`             | Normalizes incoming signal.                                                 |
| NormRange             | `normrange.gin`             | Scales the signal such that its peak equals a given value.                  |
| PeakingFilter         | `peakingfilter.gin`         | Applies random peak filtering, superposed to data.                          |
| PhaseMangle           | `phasemangle.gin`           | shuffle phase with an all-pass filter                                       |
| PitchShift            | `pitchshift.gin`            | Random pitch-shifting of the signal (duration preserved).                   |
| PolarityInversion     | `polarityinversion.gin`     | Randomly inverses the waveform (*-1)                                        |
| PreEmphasis           | `preemphasis.gin`           | adds pre-emphasis to a signal.                                              |
| RandomDelay           | `randomdelay.gin`           | random short comb-filtered delays                                           |
| RandomDistort         | `randomdistort.gin`         | random Eqing and distortion of signal                                       |
| RandomEQ              | `randomeq.gin`              | random EQing of the signal                                                  |
| Resample              | `resample.gin`              | resampling of the signal.                                                   |
| Reverse               | `reverse.gin`               | randomly reverses temporally the signal.                                    |
| RoomSimulator         | `roomsimulator.gin`         | performs room reverberation simulation.                                     |
| SevenBandParametricEQ | `sevenbandparametriceq.gin` | Random seven-band parametric EQ.                                            |
| Shift                 | `shift.gin`                 | Random time shift of the signal.                                            |
| TanhDistortion        | `tanhdistortion.gin`        | Performs random tanh distortion to the signal.                              |
| TimeMask              | `timemask.gin`              | Randomly masks temporal zones of the signal with silence.                   |
| TimeStretch           | `timestretch.gin`           | Randomly time-stretches the incoming signal, and crop.                      |
| Stretch               | `stretch.gin`               | Random resampling with polyphase filtering (changes speed and pitch).       |
