# 6. Pre-processing

Online datasets decode audio during training. For large datasets or long trainings, compiling the dataset in an LMDB database makes loading much faster : chunks, metadata and features are computed once and stored.

## From an online dataset

Once its settings are fixed, an online dataset can be compiled with the same files, settings, features and partitions :

```python
from acids_dataset.datasets import AudioDataset
from acids_dataset.features import Loudness

dataset = AudioDataset("tutorial_dataset", sr=22050, chunk_length=1., meta_regexp=["**/{{instrument}}_*.wav"],
                       features=[Loudness(sr=22050, keep_channels=False)], output_pattern="{waveform,loudness}")
dataset.split({"train": 0.8, "valid": 0.2}, seed=0, write="split")
compiled = dataset.preprocess("tutorial_compiled", force=True)
print(type(compiled).__name__, len(compiled), compiled.load_partition("split").keys())
```

## From the command line or Python

Datasets can also be compiled directly, with a [preset](../guide/presets.md) of settings :

```bash
acids-dataset preprocess --path tutorial_dataset --out tutorial_compiled --sample_rate 22050 --chunk_length 22050 \
    --feature loudness --meta_regexp "**/{{instrument}}_*.wav" --force
```

```python
from acids_dataset import preprocess_dataset

preprocess_dataset("tutorial_dataset", out="tutorial_compiled", sample_rate=22050, chunk_length=22050,
                   features=["loudness"], meta_regexp=["**/{{instrument}}_*.wav"], force=True)
```

## Loading a compiled dataset

`AudioDataset` recognizes compiled datasets, and returns a `PreprocessedAudioDataset` with the same interface :

```python
compiled = AudioDataset("tutorial_compiled", output_pattern="{waveform,loudness}")
print(type(compiled).__name__, compiled.metadata["sr"], compiled.chunk_length)
print({k: tuple(v.shape) for k, v in compiled[0].items()})
```

## Checking datasets

`info` describes datasets, and `--health` reports problematic files (clipping, DC offset, silence) :

```bash
acids-dataset info --path tutorial_compiled --files --health
```

```python
from acids_dataset.health import dataset_health

report = dataset_health(compiled)
print([(f["audio_path"], f["flags"]) for f in report["files"] if f["flags"]])
```

Here, the drums files are flagged as *mostly silent* : their bursts are separated by silence. With online datasets, `loudness_threshold` draws silent chunks again ; with `preprocess`, `exclude_flags` excludes flagged files.

Next : [training](07-training.md).

!!! info "See also"
    [Pre-processing](../guide/preprocessing.md) and [Dataset browser and health](../guide/browser.md) in the documentation.
