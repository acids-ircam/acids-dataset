# Datasets

Datasets give items (chunks of audio, with features and metadata) and access to the files they come from. `AudioDataset` returns an `OnlineAudioDataset` for raw audio folders, a `PreprocessedAudioDataset` for compiled datasets, and a `CombinedAudioDataset` for lists containing compiled datasets.

## Loading a dataset

`AudioDataset` is the common interface of datasets : instantiating it with a path returns a `PreprocessedAudioDataset` if the path is a pre-processed dataset folder (see [Pre-processing](preprocessing.md)), or an `OnlineAudioDataset` otherwise. Both share the same indexing, partitioning and loading interface.

```python
from acids_dataset.datasets import AudioDataset

dataset = AudioDataset("path/to/preprocessed")   # -> PreprocessedAudioDataset
dataset = AudioDataset("path/to/raw/audio")      # -> OnlineAudioDataset
dataset[0] # will provide automatically the "waveform" field as a single tensor
```

### Online mode

In online mode, audio files are parsed on the fly : there is no compilation step, and features are computed dynamically when data is loaded. Parsing parameters can be given directly, or with a [preset](presets.md) (explicit arguments override the preset).

```python
from acids_dataset.datasets import AudioDataset
from acids_dataset.features import Loudness

dataset = AudioDataset(
  "path/to/your/dataset",            # or a list of folders
  sr = 44100,                        # target sampling rate (None to keep original ones)
  channels = 1,                      # target number of channels (None to keep original ones)
  chunk_length = 131072,             # in samples (int) or seconds (float)
  batch_by = "auto",                 # "files", "dataset" or "auto" (see Data loaders)
  pad_mode = "zero",                 # padding for chunks exceeding files ("zero", "repeat", "reflect", "replicate")
  features = [Loudness(), "mel"],    # computed dynamically ; objects or feature config names
  meta_regexp = ["**/{{instrument}}_*.wav"], # automatically create a "instrument" feature if audio file matches the regexp
  filters = [], 
  exclude = ["**/*.opus"]
)
loader = dataset.get_loader(batch_size=16, num_workers=4)
```

At initialization, audio files are scanned to retrieve their length, sampling rate, number of channels and codec, without decoding them ; path-based metadata (`meta_regexp`, see [Features](features.md)) is also extracted. This information is cached in a `.ad` index file written in each dataset folder, such that next initializations only scan new or modified files (deleted files are removed from the index). Unreadable files are listed in `dataset.invalid_files` and are not scanned again until they are modified.

| Option        | Effect                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------ |
| `index_dir`   | write index files as `<folder_name>.ad` in this folder (useful for read-only datasets)     |
| `rescan`      | ignore existing index files, and scan every file                                           |
| `write_index` | if `False`, do not write index files                                                       |

Several dataset folders can be given, as long as they have different names. Audio files are partially loaded (only the requested chunk is decoded), resampled to `sr`, and adapted to `channels`.

### Loading audio from files

Both modes give access to the audio files of the dataset, independently of dataset items : `dataset.files` lists the files (with their `audio_path`, `duration`, `sr` and `channels`), and audio or fragments can be loaded from a given file :

```python
wav = dataset.load_audio(file_idx, start=1.0, length=2.0)        # tensor (channels, samples), at the sampling rate of the dataset
fragment = dataset.get_fragment(file_idx, start=1.0, length=2.0)  # AcidsFragment, with waveform, metadata and features
fragment.get_array("loudness")
```

Features of fragments are computed on the loaded audio (the features of the dataset, or the ones given with `features=[...]`). For pre-processed datasets, audio is rebuilt from the chunks stored in the database : parts of files that were not stored (e.g. discarded chunks) are filled with zeros, and datasets pre-processed without waveform cannot load audio.

### Baked features

Features of online datasets are computed every time a chunk is loaded. Heavy features (e.g. spectrograms, pitch, neural embeddings) can be *baked* : computed once on whole audio files, and stored in `.ad` files that chunks read instead of computing features.

```python
dataset = AudioDataset("path/to/raw/audio", sr=44100, chunk_length=131072, features=[Mel(), F0(mode="pyin"), Loudness()])
dataset.bake_features("mel_128", "f0")                  # every feature if no name is given ; path=... for another destination

# later, in another script : baked features replace the features of the same name
dataset = AudioDataset("path/to/raw/audio", sr=44100, chunk_length=131072, features=[Mel(), F0(mode="pyin"), Loudness()])
dataset.use_baked_features("mel_128", "f0")             # check=True : every file must have an up-to-date baked file
loader = dataset.get_loader(batch_size=16, num_workers=4)
```

- Baked files mirror the architecture of dataset folders : `<dataset folder>/ad-features/<relative audio path>.ad` by default, or `<path>/<dataset folder name>/<relative audio path>.ad` with `path`. Each file contains the features of the whole audio file (time last), and its metadata : original sampling rate of the file, sampling rate and channels features were computed with, frame rate (`feature_sr`) and frame offset of each feature, configuration of features, and size / modification date of the audio file. Files are written in numpy's `.npz` format (`acids_dataset.datasets.baked.read_baked_file` reads them).
- Baking is incremental : features already baked with the same configuration, for unchanged audio files, are skipped (unless `overwrite=True`), and new features are added to existing files. Partitions bake their files only.
- `use_baked_features(*names, path=None, check=True)` reads given features from baked files (every feature baked for the first file if no name is given). Dataset features of the same names are replaced, and other names are added to the features of the dataset : baked features do not need to be given to the dataset. With `check=True`, every file is checked (existing baked file, unchanged audio file, same sampling rate and channels, same configuration as the dataset feature of the same name) ; with `check=False`, missing baked files raise errors when loading. `dataset.baked_features` lists baked features.
- Chunks get the frames of their position in the features of the whole file : from the frame closest to their start, as many frames as the feature gives on a chunk. Values may thus differ from features computed on chunks by the alignment of frames (up to half a hop), and at the edges of chunks (features of chunks are computed on padded chunks, baked features on whole files).
- Only features with a time axis (last) can be baked : per-chunk descriptors (e.g. `Tempo`, `AttackTime`) and non-array features (`AfterMIDI`, metadata) cannot.

### From online to pre-processed datasets

Once settled, an online dataset can be compiled into a pre-processed dataset with the same files (filters, exclusions, without the unreadable files of the index), parsing settings (sampling rate, channels, chunk length, padding mode, `loudness_threshold`), `meta_regexp` and features :

```python
dataset = AudioDataset("path/to/raw/audio", preset="rave", meta_regexp=[...], features=[Loudness()], exclude=["**/drafts/*"])
dataset.split({"train": 0.8, "valid": 0.2}, features=["original_path"], seed=0, write="split")
preprocessed = dataset.preprocess("path/to/compiled", exclude_flags=["silent", "clipping"])   # a PreprocessedAudioDataset
partitions = preprocessed.load_partition("split")                                           # same files as online partitions
```

- `hop_length` sets the hop between pre-processed chunks (half of `chunk_length` by default), `features` other features, `waveform=False` stores features only, and `preset` / `override` other parsing parameters. The size of the database is estimated from the durations of the index.
- Written partitions are carried to the pre-processed dataset (`partitions=True`, a list of names, or `None`), matching files by audio paths : validation files stay the same in both modes.
- `exclude_flags` excludes files flagged by the [health report](browser.md#dataset-health) (computed on every file, with `health_thresholds`).
- The source of the dataset (folders, index, filters, `meta_regexp`, excluded and unreadable files) is recorded in the `source` entry of its metadata. Partitions of online datasets can be compiled too (only their files are parsed).

## Metadata

Dataset metadata is accessible through `dataset.metadata` (for pre-processed datasets, see also [Get information](preprocessing.md#get-information)). In online mode, the list of scanned files with their header information and path-based metadata is available through `dataset.files`.

`acids_dataset` also provides handy methods to retrive data from a given label :
```python
note_a_data = dataset.query_from_features(..., pitch='A') # retrieve all the data with 0 pitch class
note_a_data = dataset.query_from_features(10, pitch='A') # retrieve 10 exemples with 0 pitch class
note_a_data = dataset.query_from_features(10, pitch='A', original_path=['file1.wav', 'file2.wav']) # retrieve 10 exemples with 0 pitch class in given data files
```

Features available for querying are hashed in `dataset.feature_hash` (`{feature: {value: indices}}`). In online mode, only path-based features (`original_path` and `meta_regexp` features) can be used for search and partitioning, as they are known without loading audio. `index_for_features` then returns file indices, and `query_from_features` returns chunks of the matching files.

## Multi-dataset

Several datasets (pre-processed or online) can be combined with `CombinedAudioDataset`, that is an `AudioDataset` : items, slices, `files`, `load_audio`, `get_fragment`, `get_loader`, partitions and `to` work as for other datasets. Combined datasets must share the same output pattern. Datasets can be given as objects or as paths ; a list of paths that contains a pre-processed dataset given to `AudioDataset` also returns a combined dataset (a list of raw folders gives a single online dataset).

```python
from acids_dataset.datasets import AudioDataset, CombinedAudioDataset

dataset = CombinedAudioDataset([AudioDataset("path/to/dataset1"), AudioDataset("path/to/dataset2")], weights=[0.7, 0.3])
dataset = AudioDataset(["path/to/preprocessed", "path/to/raw/audio"], sr=44100)    # dataset arguments are given to each dataset
loader = dataset.get_loader(batch_size=16, chunk_length=131072)                     # loading arguments configure online datasets
```

`weights` are the sampling probabilities of datasets (here, 70% of items come from the first dataset, whatever the sizes of datasets) ; without weights, items are sampled uniformly. Loading arguments (`chunk_length`, `batch_by`, `features_after_transforms`...) configure the datasets they apply to ; a `chunk_length` different from the one of a pre-processed dataset raises an error, as items could not be batched together.

Combined datasets can be partitioned with `split` and `load_partition` (each dataset is split with the same arguments, and partitions are combined datasets of the partitions of each dataset). You can also parse several data folders into a single pre-processed dataset (see [Simple parsing](preprocessing.md#simple-parsing)), or give several folders to an online dataset.
