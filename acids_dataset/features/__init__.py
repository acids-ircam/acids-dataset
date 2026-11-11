import os
RAISE_EXC_IF_FEATURE_ERROR = os.environ.get('ACIDSTRANSFORMS_RAISE_EXC_IF_FEATURE_ERROR', '1').lower() in ['1', 'true', 'yes']

# features
from .base import AcidsDatasetFeature, check_feature_configs
from .regexp import RegexpFeature, append_meta_regexp, parse_meta_regexp
from .mel import Mel
from .cqt import CQT
from .loudness import Loudness
from .midi import AfterMIDI
from .module import *
from .beat_tracking import BeatTrack
from .pitch import Pitch, F0
from .spatial import Width, SpatialCovariance, CrossSpectralMatrix, SpectroSpatialCovarianceVector, InterChannelLevelDifference, InterChannelTimeDifference, InterChannelCoherence, GeneralizedCrossCorrelation, Pan
from .spectral import *
from .temporal import *
from .onsets import *
from .harmonic import *
from .quality import *
from .representations import STFT, DGT, FFT, DCT, Gammatone, Cepstrum
from .pqmf import PQMF
from .lpc import LPC
from .nn_modules import *

# advanced operations
from .clustering import hash_from_clustering
