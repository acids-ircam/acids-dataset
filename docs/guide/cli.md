# Command line

`acids-dataset` provides the following commands (`acids-dataset <command> --help` lists their options) :

| Command | Description | See |
| ------- | ----------- | --- |
| `preprocess` | compiles audio folders into a pre-processed dataset | [Pre-processing](preprocessing.md) |
| `update` | adds features or audio files to a pre-processed dataset | [Updating a compiled database](preprocessing.md#updating-a-compiled-database) |
| `add_embedding` | adds embeddings of a TorchScript model to a pre-processed dataset | [Computing embeddings](preprocessing.md#computing-embedding-from-a-torchscript-model) |
| `info` | describes a pre-processed dataset or a raw audio folder, with `--health` reports | [Get information](preprocessing.md#get-information) |
| `presets` | lists available presets, or prints one with `--preset` | [Presets](presets.md) |
| `ui` | launches the dataset browser | [Dataset browser](browser.md) |

Commands can also be run as modules, e.g. `python -m acids_dataset.ui --path ...`.

## Examples

```bash
# compile a dataset for RAVE, with loudness
acids-dataset preprocess --path audio/ --out compiled/ --preset rave --feature loudness

# metadata from file names, and selection of files
acids-dataset preprocess --path audio/ --out compiled/ --chunk_length 131072 \
    --meta_regexp "**/{{instrument}}_*.wav" --filter "**/stems/*" --exclude "**/*.opus"

# describe a raw folder (without writing its index), and check its health
acids-dataset info --path audio/ --nowrite_index --health

# add a feature to a compiled dataset
acids-dataset update --path compiled/ --feature mel

# explore a dataset in the browser
acids-dataset ui --path compiled/
```
