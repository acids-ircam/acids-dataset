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
| ApplyImpulseResponse  | `applyimpulseresponse.gin`  | Applies impulse response (needs audio files as first argument).             |
| BandPassFilter        | `bandpassfilter.gin`        | Random band-pass filter.                                                    |
| BasicPitch            | `basicpitch.gin`            | Pitch extraction based on BasicPitch                                        |
| Bitcrush              | `bitcrush.gin`              | Adds random bit-crushing (bit & sr reduction)                               |
| Clip                  | `clip.gin`                  | Clips incomping signal.                                                     |
| ClippingDistortion    | `clippingdistortion.gin`    | Adds clipping distortion.                                                   |
| Compress              | `compress.gin`              | Compression with random parameters.                                         |
| Crop                  | `crop.gin`                  | Random cropping to fixed chunk size.                                        |
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
| Mp3Compression        | `mp3compression.gin`        | Applies random mp3 compression.                                             |
| Mute                  | `mute.gin`                  | Mutes incoming incoming signal.                                             |
| Normalize             | `normalize.gin`             | Normalizes incoming signal.                                                 |
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
