# Pre-processing

Compiled datasets are stored in [LMDB](https://openldap.org/) databases, with their chunks, metadata and features. They load much faster than online datasets, and are recommended for long trainings.

## Simple parsing
`acids-dataset` is available as a command-line tool to easily pre-process a dataset path. For example, you can parse a dataset in 2048-sized chunks with the following command :

```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --chunk_length 2048
```

the `--preset` argument (or its alias `--config`) allows you to provide a [preset](presets.md) for dataset parsing. Some presets are provided in `acids_dataset/configs`, but you can also link your own by placing them into a folder `ad_configs` in the current execution directory. For exemple, you can parse a dataset for [RAVE](http://github.com/acids-ircam/RAVE) with :

```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --preset rave 
```

or, similarly, for [AFTER](http://github.com/acids-ircam/AFTER) :

```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --preset after 
```

Options given in command line (`--chunk_length`, `--sample_rate`, `--channels`...) override the values of the preset (see [Presets](presets.md)).

You can also parse several data folders to join them in a single dataset by providing several `--path` keywords :

```bash
acids-dataset preprocess --path /path/to/your/dataset1 --path /path/to/your/dataset2 --out /target/path/for/preprocessed 
```

`acids-dataset` will keep track of the original folder for each data chunk, so no information will be lost along compilation!

##### Important : the `max_db_size` is very important, as it allows you to set the maximum size of the compiled library. This is imposed by LMDB, and may be a little tricky to deal with. For little datasets (like <20Gb), you may want to add the `--compact` option that will first make a "full" compiled version of the database in a temporary folder, and then compact it in the target directory. For big datasets, be careful on how much data you allocate!

## Data selection

`acids-dataset` also allows to very easily select and exclude files inside a given folder. For exemple, here `--filter` allows to only select the files contained in a sub-dataset of the folder provided with `path`, and `--exclude` to exclude all .opus file.

```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --filter "**/*" --exclude "*.opus"
```

that would only retain audio files in subfolders of the root, and exclude all .opus files. All the preprocess options are available by executing the command with the `--help` tag :

```bash
USAGE: acids_dataset/preprocess.py [flags]
flags:

acids_dataset/preprocess.py:
  --channels: number of audio channels (overrides preset)
    (an integer)
  --[no]check: has interactive mode for data checking.
    (default: 'true')
  --chunk_length: number of samples per chunk (overrides preset)
    (an integer)
  --[no]compact: create compact version of lmdb database
    (default: 'false')
  --config: dataset preset (name or path to a .gin file)
    (default: 'default')
  --device: device for feature computation
    (default: 'cpu')
  --exclude: wildcard to exclude target files;
    repeat this option to specify a list of values
    (default: '[]')
  --feature: config files;
    repeat this option to specify a list of values
    (default: '[]')
  --filter: wildcard to filter target files;
    repeat this option to specify a list of values
    (default: '[]')
  --[no]force: force dataset preprocessing if folder already exists
    (default: 'false')
  --log: log file
  --max_db_size: maximum database size
    (default: '100.0')
    (a number)
  --meta_regexp: additional regexp for metadata parsing;
    repeat this option to specify a list of values
    (default: '[]')
  --out: parsed dataset location
  --override: additional overridings for configs;
    repeat this option to specify a list of values
    (default: '[]')
  --path: dataset path;
    repeat this option to specify a list of values
  --preset: an alias for --config
  --sample_rate: sample rate (overrides preset)
    (an integer)
  --[no]waveform: store waveform in the database (--nowaveform to only store features)
    (default: 'true')
```

You can also declaratively use this package in Python using the `preprocess_dataset` function of the `acids_dataset` package:

```python
import acids_dataset

acids_dataset.preprocess_dataset(
        dataset_path,
        out = out_path, 
        preset = "default", 
        check = False,
        chunk_length = 131072,
        sample_rate = 44100,
        channels = 1,
        flt = [], 
        exclude = [],
        waveform = True
    )
```

### Dataset health

`acids-dataset info --path ... --health` (or `--health_max_files 100`) reports the health of the files of a dataset, pre-processed or not : peak and RMS levels, ratio of clipped samples, DC offset and ratio of silent frames, with flags for silent, mostly silent, clipping, DC offset and very quiet files. The report is also available in the [dataset browser](browser.md) (Overview tab), and in Python with `acids_dataset.health.dataset_health(dataset, max_files=None, thresholds=None)`.

### Unreadable files

Files that cannot be read (corrupted, empty...) are skipped with a warning, and do not stop the parsing. Skipped files are listed in the `skipped_files` entry of the dataset metadata (see [Get information](#get-information)). If you prefer the parsing to stop at the first unreadable file, set the `ACIDSTRANSFORMS_RAISE_EXC_IF_WRITER_ERROR` environment variable to `1` (or set `acids_dataset.writers.RAISE_EXC_IF_WRITER_ERROR = True` in Python).

## Embedding features

`acids_dataset` has specific configuration files to embed metadata (like audio descriptors) in the database, to make them accessible during training. For example, to add loudness and mel profiles for each data chunk, you may add the `mel` and `loudness` configs, that are contained in `acids_dataset/configs/features`

```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --feature loudness --feature mel
```

of course, you can customize these features ; please see the section [Customizing features](extending.md#customizing-features).

The `RegexpFeature` (see [Features](features.md)) is accessible in command-line with `--meta_regexp`. With the Slakh-like example, you can extract respectively the track ids and the instrument ids into `id` and `inst` features by running :

```bash
acids-dataset preprocess --path /path/to/your/dataset --exclude "**/mix.flac" --meta_regexp "Track{{id}}/stems/S{{inst}}.flac" 
```

### Storing features only

If you only need the features during training, you can avoid storing the audio with `--nowaveform` (or `waveform=False` in `preprocess_dataset`). Features are still computed from the audio during parsing, but the waveform is not written in the database, which can drastically reduce its size. This choice is recorded in the dataset metadata (`waveform` entry), such that audio files added later with `update` follow the same setting (datasets compiled before this option are considered to contain waveforms). Such datasets only give access to their features : use a feature as `output_pattern` when loading them (see [Output format](output-patterns.md)).

```bash
acids-dataset preprocess --path /path/to/your/dataset --out /target/path/for/preprocessed --feature loudness --feature mel --nowaveform
```

## Updating a compiled database

The `update` commamnd allows to add features / files to a compiled dataset.

```bash
acids-dataset update --help

      USAGE: acids_dataset/update.py [flags]
flags:

acids_dataset/update.py:
  --[no]check: recomputes the feature if already present in the dataset
    (default: 'true')
  --data: add audio files to the target dataset.;
    repeat this option to specify a list of values
    (default: '[]')
  --exclude: wildcard to exclude target files;
    repeat this option to specify a list of values
    (default: '[]')
  --feature: add features to the target dataset.;
    repeat this option to specify a list of values
    (default: '[]')
  --filter: wildcard to filter target files;
    repeat this option to specify a list of values
    (default: '[]')
  --meta_regexp: parses additional blob wildcards as features.;
    repeat this option to specify a list of values
    (default: '[]')
  --override: add audio files to the target dataset.;
    repeat this option to specify a list of values
    (default: '[]')
  --[no]overwrite: recomputes the feature if already present in the dataset, and overwrites existing files
    (default: 'false')
  --path: dataset path
```

For example, if you want to update a pre-processed dataset with the `loudness` feature and add some more data contained in `path/to/additional/data`, you can use

```bash
acids-dataset update --path path/to/preprocessed --data path/to/additional/data --feature features/loudness
```

or, alternatively, use the python high-level function `update_dataset`

```python
from acids_dataset import update_dataset
from acids_dataset.features import Loudness

update_dataset(
  "path/to/preprocessed", 
  data=["path/to/additional/data"],
  features=[Loudness()]
)
```

##### By default, `update` will open the database with 1.1 * size of the database. If you plan to add havy audio / features, do not forget to also update `--max_db_size`!!

Audio files added with `--data` follow the `waveform` setting of the compiled dataset (see [Storing features only](#storing-features-only)), and unreadable files are skipped and appended to `skipped_files`.

## Computing embedding from a TorchScript model

A special script, `add_embedding`, provides a very handy way to compute embeddings from a scripted PyTorch model. For example, if you have a RAVE .ts file, you can populate the latent embeddings and  in your database through cuda (replace by `cpu` or `mps` in case) using:

```bash
acids-dataset add_embedding --path replace/path/to/db --module replace/path/to/ts --method encode --device cuda:0
```

An interesting feature of `add_embedding` is also the ability to compute "augmented“ embeddings, batching original data with transforms that can augment the data. This is typically very useful for self-supervised learning.  

```bash
acids-dataset add_embedding --path replace/path/to/db --module replace/path/to/ts --transform pitchshift --transform pitchshift --transform pitchshift --method encode --device cuda:0
```

Similarly, a python backend is provided

```python
from acids_dataset import add_embedding_to_dataset, transforms as adt

# load your model here...
model = Model()

# you can also directly populate with additional transforms. 
# different transforms will be batched during import. 

transforms = [adt.PitchShift() for i in range(4)]

add_embedding_to_dataset(
  path="path/to/dataset", 
  module=model, 
  module_path=None, # use path to directly load a .ts
  transforms=transforms, 
  module_sr=None, # acids_dataset will try to retrieve the sr attribute ; if not, provide it to avoid problems
  device="cpu"
)
```

## Get information

You can quickly monitor the content of a parsed metadata using the `info` metadata : 
```bash
    USAGE: acids_dataset/info.py [flags]
flags:

acids_dataset/info.py:
  --[no]check_metadata: check metadata discrepencies in dataset
    (default: 'false')
  --exclude: raw audio folders : wildcard to exclude target files;
    repeat this option to specify a list of values
    (default: '[]')
  --[no]files: list files within dataset
    (default: 'false')
  --[no]health: health report of files (clipping, DC offset, silence, level)
    (default: 'false')
  --health_max_files: maximum number of files of the health report
    (an integer)
  --filter: raw audio folders : wildcard to filter target files;
    repeat this option to specify a list of values
    (default: '[]')
  --idx: parses fragment with idx %s (file index for raw audio folders);
    repeat this option to specify a list of values
    (default: '[]')
    (an integer)
  --index_dir: raw audio folders : folder of index files (default : in the dataset folder)
  --key: parses fragment with key %s (audio path for raw audio folders);
    repeat this option to specify a list of values
    (default: '[]')
  --meta_regexp: raw audio folders : regexp for metadata parsing;
    repeat this option to specify a list of values
    (default: '[]')
  --path: dataset path (pre-processed dataset, or raw audio folder)
  --[no]rescan: raw audio folders : ignore existing index and re-scan every file
    (default: 'false')
  --[no]write_index: raw audio folders : write the .ad index file (asked if not given)
```

You can also add the `--files` to list all the parsed audio files, `--check_metadata` flag to check missing metadata, or either `--idx` or `--key` to get information for a specific sub-item. See the full available options by running `acids-dataset info --help`. For example, it should provide you informations as follows :

```bash
channels: 1
chunk_length: 1.0
exclude: []
features: {'f0': 'F0(mode=yin, sr=44100)', 'pitch': 'Pitch(mode=yin, sr=44100)'}
filters: []
format: int16
fragment_class: AcidsFragment
hop_length: 2.0
import_backend: 0
loudness_threshold: None
max_db_size: 5.1000000000000005
n_chunks: 40
n_seconds: 40.0
pad_mode: 0
parser_class: RawParser
skipped_files: []
sr: 44100
waveform: true
writer_class: LMDBWriter
```

`info` also works on raw (not pre-processed) audio folders, that are then scanned as [online datasets](datasets.md#online-mode) : it gives the number of files, total duration, sampling rates, channels, codecs, file durations and unreadable files of the folder (and `meta_regexp` values with `--meta_regexp`). `--files` lists the scanned files, and `--idx` / `--key` describe a file from its index or its audio path. If the folder has no `.ad` index yet, `info` asks whether to write it (making next scans faster) or to only scan the folder in memory ; use `--write_index` or `--nowrite_index` to answer in advance (the index is not written in non-interactive sessions). Existing indexes are updated.

```bash
acids-dataset info --path /path/to/raw/audio --meta_regexp "**/{{instrument}}_*.wav" --nowrite_index
```
