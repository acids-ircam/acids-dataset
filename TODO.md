- transforms altering pre-processed features : transforms could declare which features they alter (e.g. Gain alters 
  loudness, PitchShift alters f0 / pitch), such that pre-processed features of transformed items are recomputed (or 
  updated) instead of being read from the database. Batch features of online datasets are already computed after 
  transforms (see AudioDataset.get_loader(batch_features=...)).
- test_transforms.py has known failures (18 cases : missing optional dependencies such as cylimiter, torchaudio.sox_effects 
  removed from torchaudio, numpy / torch type handling of some transforms) ; it is excluded from continuous integration 
  (.github/workflows/tests.yml) until fixed.

- using a representation learning to get similarity embedding of the dataset to favour diversity in trainings for curriculum