import pytest
import torch, torchaudio
import random
import os
from pathlib import Path
import shutil
import gin
from . import OUT_TEST_DIR, test_name, get_feature_configs, CURRENT_TEST_DIR
import lmdb, random
from .datasets import get_available_datasets, get_dataset

import acids_dataset as ad
from acids_dataset import get_fragment_class, feature_from_gin_config
from acids_dataset.datasets import AudioDataset
from acids_dataset.utils import set_gin_constant, feature_from_gin_config
from acids_dataset.writers import LMDBWriter, read_metadata
from acids_dataset.features import Mel, Loudness, AfterMIDI, BeatTrack, hash_from_clustering

from tests.module_tests import *


def init_gin_constants():
    set_gin_constant('SAMPLE_RATE', 44100)
    set_gin_constant('CHANNELS', 1)
    set_gin_constant('CHUNK_LENGTH', 131072)
    set_gin_constant('HOP_LENGTH', 65536)
    set_gin_constant('DEVICE', 'cpu')

@pytest.mark.parametrize('config', ['default.gin'])
@pytest.mark.parametrize("dataset", get_available_datasets())
@pytest.mark.parametrize('feature_path,feature_config', get_feature_configs('mel'))
def test_mel(config, feature_path, feature_config, dataset, test_name, test_k=10):
    init_gin_constants()
    gin.add_config_file_search_path(feature_path)
    gin.parse_config_file(config)
    gin.parse_config_file(feature_config)

    dataset_path = get_dataset(dataset)
    dataset_out = OUT_TEST_DIR / "compiled" / test_name
    if dataset_out.exists():
        shutil.rmtree(dataset_out.resolve())

    # extract mel
    mel_feature = Mel()
    out = mel_feature(torch.zeros(1, 1, 16384))

    # build dataset
    writer = LMDBWriter(dataset_path, dataset_out, features=[mel_feature])
    writer.build()

    env = lmdb.open(str(dataset_out), lock=False, readonly=True)
    fragment_class = get_fragment_class(read_metadata(dataset_out)['fragment_class'])
    mel_key = list(read_metadata(dataset_out)['features'].keys())[0]

    dataset = AudioDataset(str(dataset_out))
    random_keys = random.choices(dataset.keys, k=test_k)
    for key in random_keys:
        dataset.get(key, "waveform")
        dataset.get(key, mel_key)


@pytest.mark.parametrize('config', ['default.gin'])
@pytest.mark.parametrize("dataset", get_available_datasets())
@pytest.mark.parametrize("keep_channels", [True, False])
@pytest.mark.parametrize("frame_length", [None, 1024])
def test_loudness(config,  dataset, test_name, keep_channels, frame_length, test_k=10):
    init_gin_constants()
    gin.parse_config_file(config)

    dataset_path = get_dataset(dataset)
    dataset_out = OUT_TEST_DIR / "compiled" / test_name
    if dataset_out.exists():
        shutil.rmtree(dataset_out.resolve())

    # extract loudness 
    loudness_feature = Loudness(keep_channels=keep_channels, frame_length=frame_length)
    out = loudness_feature(torch.zeros(1, 1, 16384))

    # build dataset
    writer = LMDBWriter(dataset_path, dataset_out, features=[loudness_feature])
    writer.build()

    env = lmdb.open(str(dataset_out), lock=False, readonly=True)
    fragment_class = get_fragment_class(read_metadata(dataset_out)['fragment_class'])
    with env.begin() as txn:
        dataset_keys = list(txn.cursor().iternext(values=False))
        # pick a random item
        random_keys = random.choices(dataset_keys, k=test_k)
        for key in random_keys:
            ae = fragment_class(txn.get(key))
            audio = ae.get_audio("waveform")
            loudness = ae.get_array("loudness")
            if keep_channels: 
                if frame_length:
                    assert loudness.ndim == 2
                else:
                    assert loudness.ndim == 1
            else:
                if frame_length:
                    assert loudness.ndim == 1
                else:
                    assert loudness.ndim == 0
    
        

@pytest.mark.parametrize('config', ['default.gin'])
@pytest.mark.parametrize("dataset", get_available_datasets())
@pytest.mark.parametrize('feature_path,feature_config', get_feature_configs('midi'))
def test_after_midi(config, dataset, feature_path, feature_config, test_name, test_k=10):
    init_gin_constants()
    gin.add_config_file_search_path(feature_path)
    set_gin_constant('SAMPLE_RATE', 44100)
    set_gin_constant('CHANNELS', 1)
    gin.parse_config_file(config)

    dataset_path = get_dataset(dataset)
    dataset_out = OUT_TEST_DIR / "compiled" / test_name
    if dataset_out.exists():
        shutil.rmtree(dataset_out.resolve())
    gin.parse_config_file(feature_config)

    # extract mel
    mel_feature = AfterMIDI()
    out = mel_feature(torch.zeros(1, 1, 16384))

    # build dataset
    writer = LMDBWriter(dataset_path, dataset_out, features=[mel_feature])
    writer.build()

    env = lmdb.open(str(dataset_out), lock=False, readonly=True)
    fragment_class = get_fragment_class(read_metadata(dataset_out)['fragment_class'])
    with env.begin() as txn:
        dataset_keys = list(txn.cursor().iternext(values=False))
        # pick a random item
        random_keys = random.choices(dataset_keys, k=test_k)
        for key in random_keys:
            ae = fragment_class(txn.get(key))
            audio = ae.get_audio("waveform")
            midi = ae.get_data("midi")
    
        
@pytest.mark.parametrize('config', ['default.gin'])
@pytest.mark.parametrize("dataset", ['simple'])
@pytest.mark.parametrize('feature_path,feature_config', get_feature_configs('module'))
def test_module(config, dataset, feature_path, feature_config, test_name, test_k=10):
    init_gin_constants()
    if feature_config == "moduleembedding.gin": 
        pytest.skip(reason="moduleembedding.gin not valid for testing")
    gin.add_config_file_search_path(feature_path)
    set_gin_constant('SAMPLE_RATE', 44100)
    set_gin_constant('CHANNELS', 1)
    gin.parse_config_file(config)

    dataset_path = get_dataset(dataset)
    dataset_out = OUT_TEST_DIR / "compiled" / test_name
    if dataset_out.exists():
        shutil.rmtree(dataset_out.resolve())

    # # extract mel
    module_feature = feature_from_gin_config(feature_path / feature_config)
    out = module_feature[0](torch.zeros(1, 1, 131072))

    # # build dataset
    writer = LMDBWriter(dataset_path, dataset_out, features=module_feature)
    writer.build()

    env = lmdb.open(str(dataset_out), lock=False, readonly=True)
    loader = AudioDataset(str(dataset_out))
    random_keys = random.choices(loader.keys, k=test_k)
    for r in random_keys:
        embedding = loader.get(r, output_pattern=module_feature[0].feature_name)
        assert embedding.shape[0] == (len(module_feature[0]._transforms) + 1)

        
@pytest.mark.parametrize('config', ['default.gin'])
@pytest.mark.parametrize("dataset", ['simple'])
@pytest.mark.parametrize('feature_path,feature_config', get_feature_configs('mel'))
def test_feature_clustering(config, dataset, feature_path, feature_config, test_name, n_clusters = 3, test_k=10):
    init_gin_constants()
    gin.add_config_file_search_path(feature_path)
    gin.parse_config_file(config)
    gin.parse_config_file(feature_config)

    dataset_path = get_dataset(dataset)
    dataset_out = OUT_TEST_DIR / "compiled" / test_name
    if dataset_out.exists():
        shutil.rmtree(dataset_out.resolve())

    # extract mel
    mel_feature = Mel()

    # build dataset
    writer = LMDBWriter(dataset_path, dataset_out, features=[mel_feature])
    writer.build()

    # hash from cluster
    dataset = AudioDataset(dataset_out, required_fields=['waveform', mel_feature.feature_name])
    kmeans = hash_from_clustering(dataset, mel_feature.feature_name, n_clusters, pca_target_dim=0, write=True, verbose=True)
    
    for cluster_idx in range(n_clusters):
        data = dataset.query_from_features(..., **{mel_feature.feature_name: cluster_idx})
        data = dataset.query_from_features(..., **{mel_feature.feature_name: cluster_idx, "original_path": "simple_dataset/long/chord_tutti_simple_0.flac"})

from acids_dataset.features.pitch import _AD_F0_METHODS

def make_sin_dataset(target_dir, n_examples, duration = 4.0, fs=44100):
    os.makedirs(target_dir)
    freqs = [random.randrange(60, 880) for n in range(n_examples)]
    for f in freqs:
        t = torch.arange(int(duration * fs))
        x = 0.8 * torch.sin(2 * torch.pi * f * t / fs + random.random() * 2 * torch.pi)
        torchaudio.save(str(target_dir / ("sin_%d.wav"%f)), x[None], sample_rate=44100)

@pytest.mark.parametrize("pitch_method", _AD_F0_METHODS)
@pytest.mark.parametrize("frame_length", [None, 1024, 2048])
@pytest.mark.parametrize("hop_length", [None, 1024, 2048])
def test_pitch_features(pitch_method, test_name, frame_length, hop_length, n_examples = 20): 
    init_gin_constants()
    ad.features.RAISE_EXC_IF_FEATURE_ERROR = True

    gin.parse_config_file("default.gin")
    set_gin_constant('SAMPLE_RATE', 44100)
    set_gin_constant('CHANNELS', 1)
    set_gin_constant('DEVICE', "cpu")
    set_gin_constant('CHUNK_LENGTH', 44100)
    
    dataset_dir = CURRENT_TEST_DIR / "datasets" / "sin_dataset"

    if not dataset_dir.exists():
        make_sin_dataset(dataset_dir, n_examples)
    
    features = [f'f0_{pitch_method}.gin', f'pitch_{pitch_method}.gin']
    feature_list = []
    
    test_sizes = [44100, 48000, 131072]
    test_channels = [1, 2, 4]
    for f in features: 
        f = feature_from_gin_config(f, add_args=(tuple(), {'frame_length': frame_length, 'hop_length': hop_length}))
        feature_list.extend(f)
        for s in test_sizes: 
            for c in test_channels: 
                x = torch.zeros(1, c, s)
                out = f[0](x)
                assert out.shape[-1] == x.unfold(-1, frame_length or f[0].default_frame_length, hop_length or f[0].default_frame_length).shape[-2]

    dataset_out = OUT_TEST_DIR / "compiled" / test_name 
    writer = LMDBWriter(dataset_dir, dataset_out, features=feature_list, force=True)
    writer.build()

    
@pytest.mark.parametrize('config', ['default.gin'])
@pytest.mark.parametrize("dataset", ['simple'])
@pytest.mark.parametrize("module", [ConvEmbedding()])
@pytest.mark.parametrize('feature_path,feature_config', get_feature_configs('beattrack'))
def test_beat_tracking(config, dataset, module, feature_path, feature_config, test_name, test_k=10):
    set_gin_constant('SAMPLE_RATE', 44100)
    set_gin_constant('DEVICE', "cpu")
    set_gin_constant('CHANNELS', 1)
    gin.add_config_file_search_path(feature_path)
    gin.parse_config_file(config)
    gin.parse_config_file(feature_config)

    dataset_path = get_dataset(dataset)
    dataset_out = OUT_TEST_DIR / "compiled" / test_name
    if dataset_out.exists():
        shutil.rmtree(dataset_out.resolve())

    beat_feature = BeatTrack(downsample=module.downsample)
    out = beat_feature(torch.randn(1, 1, 441000))

    # # build dataset
    writer = LMDBWriter(dataset_path, dataset_out, features=[beat_feature])
    writer.build()

    env = lmdb.open(str(dataset_out), lock=False, readonly=True)
    fragment_class = get_fragment_class(read_metadata(dataset_out)['fragment_class'])
    with env.begin() as txn:
        dataset_keys = list(txn.cursor().iternext(values=False))
        # pick a random item
        random_keys = random.choices(dataset_keys, k=test_k)
        for key in random_keys:
            ae = fragment_class(txn.get(key))
            audio = ae.get_audio("waveform")
            beat_clock = ae.get_array(beat_feature.feature_name)

