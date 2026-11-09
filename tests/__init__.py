import pytest
import os, sys
from pathlib import Path
import gin
import re

CURRENT_TEST_DIR = Path(__file__).parent
OUT_TEST_DIR = CURRENT_TEST_DIR / "outs"
MAX_TEST_GETITEM_IDX = 100
os.makedirs(OUT_TEST_DIR, exist_ok=True)

sys.path.append(str((CURRENT_TEST_DIR / '..').resolve()))
gin.add_config_file_search_path(str((CURRENT_TEST_DIR / '..' / 'acids_dataset' / 'configs').resolve()))


@pytest.fixture
def test_name(request):
    return request.node.name

def feature_config_from_path(feature_path, feature_name):
    if not feature_path.exists():
        return []
    feature_configs = list(filter(lambda x: re.match(rf"{feature_name}.*\.gin", x) is not None, os.listdir(feature_path.resolve())))
    return [(feature_path.resolve(), fc) for fc in feature_configs]

def get_feature_configs(feature_name):
    configs = feature_config_from_path(Path(__file__).parent / ".." / "acids_dataset" / "configs" / "features", feature_name)
    configs.extend(feature_config_from_path(Path(__file__).parent / "configs" / "features", feature_name))
    return configs

def get_available_features():
    feature_path = Path(__file__).parent / ".." / "acids_dataset" / "configs" / "features"
    features = filter(lambda x: os.path.splitext(x)[1] == ".gin", os.listdir(feature_path))
    return [feature_path / f for f in features]

def check_dataset_loading(dataset_out, waveform, features=[], test_k=1):
    """checks that waveform is stored in the database only if waveform=True, and that computed 
    features can be loaded. If waveform=False, only features are loaded."""
    import random
    from acids_dataset.datasets import AudioDataset
    dataset = AudioDataset(str(dataset_out))
    feature_status = dataset.loader.feature_status
    if feature_status is not None and "waveform" in feature_status:
        assert feature_status["waveform"].count() == (len(dataset) if waveform else 0)
    metadata = dataset.metadata
    chunk_length_smp = int(float(metadata['chunk_length']) * int(metadata['sr']))
    for idx in random.choices(range(len(dataset)), k=test_k):
        assert dataset.loader[idx].has_buffer("waveform") == waveform
        if waveform:
            audio = dataset.get(idx, output_pattern="waveform")
            assert audio.shape[-1] == chunk_length_smp
        for f in features:
            dataset.get(idx, output_pattern=f)
    return dataset
