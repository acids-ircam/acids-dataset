
import os, sys
import random
import lmdb
from acids_dataset.writers.lmdb_writer import non_buffer_keys
import shutil
import torch
import torchaudio
import pytest
import gin

from . import OUT_TEST_DIR, test_name, MAX_TEST_GETITEM_IDX, check_dataset_loading
from pathlib import Path
import acids_dataset.transforms as adt
from acids_dataset.writers import audio_paths_from_dir, LMDBWriter, read_metadata
from acids_dataset.parsers import raw_parser as raw
from acids_dataset.datasets import AudioDataset, CombinedAudioDataset
from acids_dataset.utils import loudness, set_gin_constant
from acids_dataset import transforms
from acids_dataset import get_fragment_class, features, preprocess_dataset
from .datasets import get_available_datasets, get_dataset, get_available_datasets_with_filters

          
@pytest.mark.parametrize('config', ['rave.gin'])
@pytest.mark.parametrize("waveform", [True, False])
def test_preprocess_multi(config, waveform, test_name, test_k = 1):

    set_gin_constant('SAMPLE_RATE', 44100)
    set_gin_constant('CHANNELS', 1)
    datasets = get_available_datasets()
    # test writing
    gin.parse_config_file(config)
    data_paths = []
    for d in datasets:
        data_paths.append(get_dataset(d))
    dataset_out = OUT_TEST_DIR / "compiled" / test_name
    if dataset_out.exists():
        shutil.rmtree(dataset_out.resolve())

    writer = LMDBWriter(data_paths, dataset_out, waveform=waveform)
    writer.build()

    # test loading
    env = lmdb.open(str(dataset_out), lock=False, readonly=True)
    fragment_class = get_fragment_class(read_metadata(dataset_out)['fragment_class'])
    with env.begin() as txn:
        # metadata entries (features, feature hash...) are not fragments
        dataset_keys = [k for k in txn.cursor().iternext(values=False) if k.decode() not in non_buffer_keys]
        # pick a random item
        random_keys = random.choices(dataset_keys, k=test_k)
        for key in random_keys:
            ae = fragment_class(txn.get(key))
            assert ae.has_buffer("waveform") == waveform
            if waveform:
                audio = ae.get_audio("waveform")
    env.close()
    check_dataset_loading(dataset_out, waveform, test_k=test_k)


          

@pytest.mark.parametrize('datasets', [['simple', 'simple_midi']])
@pytest.mark.parametrize('config', ['rave.gin'])
@pytest.mark.parametrize("output_pattern,transforms", [
    ("waveform", []),
    ("waveform,", [adt.Gain()]),
    ("{waveform,}", {'waveform': adt.Gain()})
])
def test_combined_dataset(config, datasets, output_pattern, transforms, test_name, test_k = 1):
    # test writing
    set_gin_constant('SAMPLE_RATE', 44100)
    set_gin_constant('CHANNELS', 1)
    gin.parse_config_file(config)
    data_paths = []

    # write datasets
    for i, dataset in enumerate(datasets):
        dataset_path = get_dataset(dataset)
        dataset_out = OUT_TEST_DIR / "compiled" / f"{test_name}_{i}"
        if dataset_out.exists():
            shutil.rmtree(dataset_out.resolve())
        writer = LMDBWriter(dataset_path, dataset_out)
        data_paths.append(dataset_out)
        writer.build()

    datasets = [AudioDataset(d, output_pattern=output_pattern, transforms=transforms) for d in data_paths]
    combined_dataset = CombinedAudioDataset(datasets)

    assert len(combined_dataset) == sum([len(d) for d in datasets])
    for i in range(len(combined_dataset))[:MAX_TEST_GETITEM_IDX]:
        out = combined_dataset[i]

    # try partition
    target_partition = {'train': 0.8, 'val': 0.2}
    partitions = combined_dataset.split(partitions=target_partition, write="random")
    for i in range(len(combined_dataset))[:MAX_TEST_GETITEM_IDX]:
        out = combined_dataset[i]
    combined_dataset.load_partition("random")
    partitions = combined_dataset.split(partitions=target_partition, features=['original_path'], write="path")
    for i in range(len(combined_dataset))[:MAX_TEST_GETITEM_IDX]:
        out = combined_dataset[i]
    combined_dataset.load_partition("path")
    partitions = combined_dataset.split(partitions=target_partition, features=['original_path'], balance_cardinality=True, write="balanced_path")
    for i in range(len(combined_dataset))[:MAX_TEST_GETITEM_IDX]:
        out = combined_dataset[i]
    combined_dataset.load_partition("balanced_path")


