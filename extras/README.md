# Example notebooks

Notebooks showcasing `acids-dataset`. Every notebook generates its own synthetic audio (in a temporary folder), such that
it runs as is, on CPU, in under a minute.

```bash
pip install acids-dataset jupyter matplotlib
# the last notebook of curriculum/ also uses Lightning
pip install pytorch-lightning
```

## Features

| Notebook | Content |
| -------- | ------- |
| [01_spectral_and_temporal](features/01_spectral_and_temporal.ipynb) | spectral shape (centroid, rolloff, flatness, contrast...), temporal (RMS, attack, ZCR) and quality descriptors ; channels and batches |
| [02_loudness_pitch_harmony](features/02_loudness_pitch_harmony.ipynb) | loudness (LUFS), F0 (yin / pyin), mel and CQT spectrograms, chroma, harmonicity and inharmonicity |
| [03_onsets_and_tempo](features/03_onsets_and_tempo.ipynb) | onset strength, onset rate, tempo and its prior |
| [04_spatial](features/04_spatial.ipynb) | pan, ILD, mid / side width, ITD and generalized cross-correlation (GCC-PHAT...), direction of arrival with a microphone array, covariance and cross-spectral matrices, SSCV for ambisonics |
| [05_features_in_datasets](features/05_features_in_datasets.ipynb) | features and metadata in output patterns, queries, batch features, class balancing, pre-processing |

## Augmentations

| Notebook | Content |
| -------- | ------- |
| [01_basics](augmentations/01_basics.ipynb) | applying transforms, probabilities, `rand_batchwise`, `Compose`, gin configurations |
| [02_gallery](augmentations/02_gallery.ipynb) | spectrograms (and sounds) of the transforms, by family : dynamics, filters, noise, distortion, time and pitch, rooms and impulse responses |
| [03_augmentations_in_datasets](augmentations/03_augmentations_in_datasets.ipynb) | transforms and output patterns, multi-view outputs, order of features and transforms, batch transforms, variable lengths |

## Curriculum

| Notebook | Content |
| -------- | ------- |
| [01_balancing_and_rarity](curriculum/01_balancing_and_rarity.ipynb) | imbalanced classes : natural sampling, class balancing and rarity curriculum, with a linear classifier |
| [02_custom_difficulty](curriculum/02_custom_difficulty.ipynb) | curriculum schedules and modes, difficulty from metadata (SNR) with a tiny pitch estimator, difficulties changing over epochs, step-based curricula |
| [03_self_paced_learning](curriculum/03_self_paced_learning.ipynb) | loss feedback with `LossTracker` on a tiny auto-encoder, in a manual loop and with Lightning's `CurriculumCallback` |
