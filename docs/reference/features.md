# Features

## Base

::: acids_dataset.features.base.AcidsDatasetFeature

## Loudness, pitch and spectrograms

::: acids_dataset.features.loudness.Loudness

::: acids_dataset.features.pitch.F0

::: acids_dataset.features.pitch.Pitch

::: acids_dataset.features.mel.Mel

::: acids_dataset.features.cqt.CQT

## Spectral descriptors

::: acids_dataset.features.spectral
    options:
      show_root_heading: false
      members: [SpectralCentroid, SpectralSpread, SpectralSkewness, SpectralKurtosis, SpectralRolloff, SpectralFlatness, SpectralCrest, SpectralFlux, SpectralContrast]

## Spectral representations, filter banks and LPC

::: acids_dataset.features.representations
    options:
      show_root_heading: false
      members: [STFT, STFTModule, STFTStreamingModule, DGT, FFT, DCT, Gammatone, Cepstrum, spectrum_mode, spectrum_from_modes]

::: acids_dataset.features.pqmf
    options:
      show_root_heading: false
      members: [PQMF, pqmf_prototype, pqmf_filters]

::: acids_dataset.features.lpc
    options:
      show_root_heading: false
      members: [LPC, levinson_durbin, line_spectral_frequencies]

## Temporal descriptors

::: acids_dataset.features.temporal
    options:
      show_root_heading: false
      members: [RMS, ZeroCrossingRate, AttackTime, TemporalCentroid, CrestFactor]

## Onsets and rhythm

::: acids_dataset.features.onsets
    options:
      show_root_heading: false
      members: [OnsetStrength, OnsetRate, Tempo]

::: acids_dataset.features.beat_tracking.BeatTrack

## Harmonic descriptors

::: acids_dataset.features.harmonic
    options:
      show_root_heading: false
      members: [Chroma, Harmonicity, Inharmonicity]

## Spatial descriptors

::: acids_dataset.features.spatial
    options:
      show_root_heading: false
      members: [Width, SpatialCovariance, CrossSpectralMatrix, SpectroSpatialCovarianceVector, InterChannelLevelDifference, InterChannelTimeDifference, InterChannelCoherence, GeneralizedCrossCorrelation, Pan]

## Quality descriptors

::: acids_dataset.features.quality
    options:
      show_root_heading: false
      members: [Peak, LoudnessMeter, a_weighting, ClippingRatio, DCOffset, DynamicRange, SilenceRatio]

## Metadata and embeddings

::: acids_dataset.features.regexp.RegexpFeature

::: acids_dataset.features.module.ModuleEmbedding
