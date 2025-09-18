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
        

