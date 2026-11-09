# Presets

Presets are an advanced way to share parsing settings between pre-processing and online datasets. They are gin configuration files describing how audio is parsed : sampling rate, channels, chunking, padding, and features. The same preset can be used to pre-process a dataset or to load it online, giving the same data in both modes.

```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --preset rave
```

```python
from acids_dataset import preprocess_dataset
from acids_dataset.datasets import AudioDataset

preprocess_dataset("path/to/your/dataset", out="path/to/preprocessed", preset="rave")
dataset = AudioDataset("path/to/your/dataset", preset="rave")      # online, with the same parsing
dataset = AudioDataset("path/to/preprocessed", preset="rave")      # warns if the dataset does not match the preset
```

| Name          | Content                                                                                                                       |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `default.gin` | base preset, included by other presets. Do not forget to give `chunk_length`, as this is not provided!                        |
| `rave.gin`    | base preset for [RAVE](http://github.com/acids-ircam/RAVE)                                                                    |
| `after.gin`   | base preset for [AFTER](http://github.com/acids-ircam/AFTER) ; MIDI can be parsed by adding the `aftermidi` feature           |
| `ddsp.gin`    | base preset for DDSP models                                                                                                   |

Presets define the following values, that are overridden by the arguments given by the user (`--chunk_length`, `--sample_rate`, `--channels`, `--device` in command line ; `chunk_length`, `sample_rate`, `channels`, `device` in `preprocess_dataset` ; `chunk_length`, `sr`, `channels` for online datasets) : 

| Value          | Description                                                                         |
| -------------- | ----------------------------------------------------------------------------------- |
| `SAMPLE_RATE`  | target sampling rate                                                                |
| `CHANNELS`     | target number of channels (`None` to keep original channels)                        |
| `CHUNK_LENGTH` | chunk length, in samples (int) or seconds (float)                                   |
| `HOP_LENGTH`   | hop length between pre-processed chunks (`None` for `CHUNK_LENGTH`)                 |
| `DEVICE`       | device for feature computation (`"cpu"` by default, `"cuda"`, `"mps"` or `"auto"`)   |

Available presets and their values can be listed with `acids-dataset presets` (and `acids-dataset presets --preset rave` prints a given preset). In Python, see `acids_dataset.presets.list_presets` and `get_preset_values`.

## Writing presets

Custom presets can be placed in an `ad_configs` folder of the current working directory (local presets take precedence over the ones of `acids_dataset/configs`), or given as paths to `.gin` files. Presets usually include `default.gin`, that binds these values to pre-processing (`RawParser`) and online datasets (`OnlineAudioDataset`), and redefine them. Features of `features.parse_features` are computed in both modes ; parameters specific to one mode can be bound to `RawParser` (pre-processing, e.g. `loudness_threshold`, `discard_if_lower_than`) or to `OnlineAudioDataset` (online, e.g. `batch_by`, `pad_mode`). For example, `$(pwd)/ad_configs/rave_loudness.gin` : 

```yaml
include 'rave.gin'

SAMPLE_RATE = 48000

Loudness:
    sr = %SAMPLE_RATE
    device = %DEVICE

features.parse_features:
    features = [@features.Loudness]

RawParser:
    loudness_threshold = -60

OnlineAudioDataset:
    batch_by = "dataset"
```

Additional gin bindings can be given with `--override` in command line, `override` in `preprocess_dataset`, and `preset_overrides` for datasets :

```python
dataset = AudioDataset("path/to/your/dataset", preset="rave", preset_overrides=["CHUNK_LENGTH = 131072", "OnlineAudioDataset.batch_by = 'files'"])
```
