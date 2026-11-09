import pytest
import gin
import shutil
import acids_dataset as ad
from acids_dataset import transforms as adt
from . import OUT_TEST_DIR, test_name
from .datasets import get_available_datasets, get_dataset


@pytest.mark.parametrize("dataset", ["simple"])
@pytest.mark.parametrize("output_pattern,transforms", [
    ("waveform", []),
    ("(waveform,)", [adt.Gain()]),
    ("(waveform,)", [adt.Compose(adt.Gain(), adt.Gain())]),
    ("(waveform,waveform)", [adt.Gain(), adt.Gain()]),
    ("(waveform[0],)", [adt.Gain()]),
    ("(waveform[:1],)", [adt.Gain()]),
    ("(waveform[0:],)", [adt.Gain()]),
    ("(waveform[0:1:1],)", [adt.Gain()]),
    ("{waveform,}", {'waveform': adt.Gain()}),
    ("{waveform->z,}", {'z': adt.Gain()}),
    ("{waveform[0]->z,}", {'z': adt.Gain()}),
    ("{waveform[:1]->z,}", {'z': adt.Gain()}),
    ("{waveform[0:]->z,}", {'z': adt.Gain()}),
    ("{waveform[0:1:1]->z,}", {'z': adt.Gain()}) ,
])
def test_audio_dataset(dataset, transforms, output_pattern, test_name):
    preprocessed_path = OUT_TEST_DIR / f"{dataset}_preprocessed"
    if not preprocessed_path.exists():
        dataset_path = get_dataset(dataset)
        ad.preprocess_dataset(dataset_path, out = preprocessed_path, chunk_length=131072, sample_rate=44100)
    dataset = ad.datasets.AudioDataset(preprocessed_path, transforms, output_pattern)
    assert len(dataset) > 0, "dataset seems empty"
    for i in range(len(dataset)):
        out = dataset[i]


@pytest.mark.parametrize("dataset", get_available_datasets())
def test_dataset_partitions(dataset):
    gin.clear_config()
    preprocessed_path = OUT_TEST_DIR / f"{dataset}_preprocessed"
    if preprocessed_path.exists(): 
        shutil.rmtree(preprocessed_path)
    dataset_path = get_dataset(dataset)
    ad.preprocess_dataset(dataset_path, out=preprocessed_path, chunk_length=88200, sample_rate=44100)
    dataset = ad.datasets.AudioDataset(preprocessed_path) 

    # test random split    
    target_partition = {'train':0.8, 'test':0.2}
    partitions = dataset.split(target_partition, write="random")
    for k, v in partitions.items():
        for i in range(len(v)):
            out = v[i]
    dataset.load_partition("random", True)

    partitions = dataset.split(target_partition, features=['original_path'], write="path")
    for k, v in partitions.items():
        for i in range(len(v)):
           out = v[i]
    dataset.load_partition("path", True)

    partitions = dataset.split(target_partition, features=['original_path'], balance_cardinality=True, write="balanced_path")
    for k, v in partitions.items():
        for i in range(len(v)):
            out = v[i]
    dataset.load_partition("balanced_path", True)
        



@pytest.mark.parametrize("features", [["instrument"], ["instrument", "take"]])
@pytest.mark.parametrize("target", [{"train": 0.7, "valid": 0.3}, {"train": 0.5, "valid": 0.2, "test": 0.3}])
def test_balanced_partitions(features, target, test_name):
    import os, torch, torchaudio
    gin.clear_config()
    # unbalanced groups : piano files are long, drums are short
    data_path = OUT_TEST_DIR / "balanced" / test_name / "data"
    if data_path.parent.exists():
        shutil.rmtree(data_path.parent)
    os.makedirs(data_path)
    sr = 44100
    for inst, n_files, duration in [("piano", 6, 6), ("guitar", 4, 3), ("drums", 6, 1), ("bass", 3, 2)]:
        for i in range(n_files):
            torchaudio.save(str(data_path / f"{inst}_{i % 2}_{i}.wav"), 0.1 * torch.randn(1, duration * sr), sample_rate=sr)
    preprocessed_path = data_path.parent / "preprocessed"
    ad.preprocess_dataset(data_path, out=preprocessed_path, chunk_length=sr, sample_rate=sr, channels=1, force=True, 
                          meta_regexp=["**/{{instrument}}_*_*.wav", "**/*_{{take}}_*.wav"])
    dataset = ad.datasets.AudioDataset(preprocessed_path)
    feature_hash = dataset.feature_hash
    assert set(feature_hash['instrument'].keys()) == {"piano", "guitar", "drums", "bass"}

    partitions = dataset.split(target, features=features, balance_cardinality=True, tolerance=0.5)
    assert set(partitions.keys()) == set(target.keys())
    keys = {p: set(v._subindices) for p, v in partitions.items()}
    # every item is assigned once (ratios sum to 1)
    assert sum(len(k) for k in keys.values()) == len(dataset)
    assert set.union(*keys.values()) == set(dataset._loader._keys)
    # feature values belong to a single partition
    for label_keys in [set(map(dataset._loader.keygen.from_str, v)) for v in feature_hash[features[0]].values()] if len(features) == 1 else []:
        assert sum(len(label_keys & k) > 0 for k in keys.values()) == 1
    for p, v in partitions.items():
        assert v.is_partition
        for i in range(len(v)):
            v[i]
    # balanced partitions of a restricted dataset
    sub = partitions["train"]
    sub._parent = None
    sub_partitions = sub.split({"a": 0.5, "b": 0.5}, features=features, balance_cardinality=True, tolerance=0.5)
    assert sum(len(v) for v in sub_partitions.values()) == len(sub)
