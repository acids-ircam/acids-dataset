import os
RAISE_EXC_IF_FEATURE_ERROR = bool(os.environ.get('ACIDSTRANSFORMS_RAISE_EXC_IF_FEATURE_ERROR', True))

# features
from .base import AcidsDatasetFeature, check_feature_configs
from .regexp import RegexpFeature, append_meta_regexp, parse_meta_regexp
from .mel import Mel
from .loudness import Loudness
from .midi import AfterMIDI
from .module import *
from .beat_tracking import BeatTrack
from .pitch import Pitch, F0

# advanced operations
from .clustering import hash_from_clustering
