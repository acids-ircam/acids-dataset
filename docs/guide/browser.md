# Dataset browser and health

## Dataset browser

<center><img src="assets/web-browser.png/"></center>

`acids-dataset` provides a simple web interface to explore pre-processed datasets and raw audio folders, launched with :

```bash
acids-dataset ui --path /path/to/dataset                                      # pre-processed dataset
acids-dataset ui --path /path/to/raw/audio --feature loudness --chunk_length 1. --meta_regexp "**/{{instrument}}_*.wav"
```

or in Python with `acids_dataset.ui.launch("path/to/dataset", **dataset_kwargs)`. It opens in the browser, with the following tabs :

| Tab           | Content                                                                                                                                  |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Overview      | dataset summary (items, files, durations, sampling rates, features, partitions), list of files, and health report                         |
| Browse        | listen to dataset items (with their waveform, spectrogram, features and metadata), or to a part of a file                                 |
| Statistics    | statistics and histograms of features over random items or over the whole dataset (computed on loaded audio for online datasets), and number of items per value of categorical features |
| Augmentations | applies transforms (from `acids_dataset/configs/transforms`) to an item, to listen to and compare original and augmented audio            |
| Partitions    | creates partitions (random, or grouped by features, balanced, seeded ; previewed before being written), and shows sizes, distribution of categorical features, and files shared between partitions |

For raw audio folders, the options of online datasets are available (`--preset`, `--sample_rate`, `--chunk_length` (one second by default), `--batch_by`, `--feature`, `--meta_regexp`, `--filter`, `--exclude`, `--write_index`) ; see `acids-dataset ui --help`. The browser can also be built from an existing dataset with `acids_dataset.ui.DatasetBrowser(dataset)` and `acids_dataset.ui.build_app`.

## Dataset health

`acids-dataset info --path ... --health` (or `--health_max_files 100`) reports the health of the files of a dataset, pre-processed or not : peak and RMS levels, ratio of clipped samples, DC offset and ratio of silent frames, with flags for silent, mostly silent, clipping, DC offset and very quiet files. The report is also available in the [dataset browser](#dataset-browser) (Overview tab), and in Python with `acids_dataset.health.dataset_health(dataset, max_files=None, thresholds=None)`.
