import os, sys
import re
import random
import lmdb
import shutil
import torch
import torchaudio
import pytest
import gin

from . import OUT_TEST_DIR, test_name, get_available_features, check_dataset_loading
from pathlib import Path
from acids_dataset.writers import audio_paths_from_dir, LMDBWriter, LMDBLoader, read_metadata
from acids_dataset.parsers import raw_parser as raw
from acids_dataset.datasets import AudioDataset
from acids_dataset.utils import loudness, GinEnv, feature_from_gin_config, set_gin_constant
from acids_dataset import transforms as adt, ACIDS_DATASET_CONFIG_PATH
from acids_dataset import get_fragment_class, features, update_dataset, add_embedding 
from .datasets import get_available_datasets, get_dataset


class ModuleTest(torch.nn.Module):
    def forward(self, x): 
        return x[..., ::32]

additional_args = {
     "beattrack": (tuple(), {"downsample": 2048}),
     "regexpfeature": (("{{NAME}}",), {"name": "filename"}),
     "moduleembedding": ((ModuleTest(),), {})
}

# features defined on stereo signals only, computed on stereo datasets
stereo_features = {"midside"}


@pytest.mark.parametrize('config', ['rave.gin'])
@pytest.mark.parametrize("dataset", ["simple"])
@pytest.mark.parametrize("feature", get_available_features())
def test_update_dataset_features(config, dataset, feature, test_name, test_k = 1):
    gin.clear_config()
    # test writing
    set_gin_constant('SAMPLE_RATE', 44100)
    set_gin_constant('CHANNELS', 2 if feature.stem in stereo_features else 1)
    set_gin_constant('DEVICE', "cpu")
    gin.parse_config_file(config)
    dataset_path = get_dataset(dataset)
    dataset_out = OUT_TEST_DIR / "compiled" / test_name
    if dataset_out.exists():
        shutil.rmtree(dataset_out.resolve())

    writer = LMDBWriter(dataset_path, dataset_out)
    writer.build()

    # pitch features require optional backends (torchcrepe, pesto, pyworld...)
    from acids_dataset.features.pitch import _AD_F0_METHODS
    pitch_mode = re.match(r"^(?:f0|pitch)_(\w+)$", feature.stem)
    if pitch_mode is not None and pitch_mode.groups()[0] not in _AD_F0_METHODS:
        pytest.skip(f"pitch mode {pitch_mode.groups()[0]} not available")

    # create feature
    add_args = additional_args.get(feature.stem, (tuple(), dict()))
    with GinEnv(paths=ACIDS_DATASET_CONFIG_PATH):
        features = feature_from_gin_config(feature, add_args=add_args)
    update_dataset(dataset_out, features, overwrite=True)

    # test loading
    loader = writer.loader(dataset_out)
    for k in loader.iter_fragment_keys():
        for f in features:
            f.read(loader[k])



@pytest.mark.parametrize('config', ['rave.gin'])
@pytest.mark.parametrize("dataset", ["simple"])
@pytest.mark.parametrize("module_path,transforms", [("tests/scripted/conv_embedding.ts", []), 
                                                    ("tests/scripted/conv_embedding.ts", [adt.Gain(), adt.AddNoise()])])
def test_add_embedding(config, dataset, module_path, transforms, test_name, test_k = 1):
    gin.clear_config()
    # test writing
    set_gin_constant('SAMPLE_RATE', 44100)
    set_gin_constant('CHANNELS', 1)
    set_gin_constant('DEVICE', "cpu")
    gin.parse_config_file(config)
    dataset_path = get_dataset(dataset)
    dataset_out = OUT_TEST_DIR / "compiled" / test_name
    if dataset_out.exists():
        shutil.rmtree(dataset_out.resolve())

    writer = LMDBWriter(dataset_path, dataset_out)
    writer.build()

    # create feature
    embedding_name = os.path.splitext(os.path.basename(module_path))[0]
    add_embedding.add_embedding_to_dataset(dataset_out, module_path=module_path, transforms=transforms, overwrite=True, name = embedding_name)

    # test loading
    loader = writer.loader(dataset_out)
    for i, k in enumerate(loader.iter_fragment_keys()):
        embedding = loader[k].get_array(embedding_name)
        assert (embedding.shape[0] - 1) == len(transforms)
        if i > test_k: break

    


@pytest.mark.parametrize('config', ['rave.gin'])
@pytest.mark.parametrize("dataset,additional_dataset", [("simple_midi", "simple_midi")])
@pytest.mark.parametrize("waveform", [True, False])
def test_update_dataset_data(config, dataset, additional_dataset, waveform, test_name, test_k = 10):
    gin.clear_config()
    set_gin_constant('SAMPLE_RATE', 44100)
    set_gin_constant('CHANNELS', 1)
    set_gin_constant('DEVICE', "cpu")
    gin.parse_config_file(config)
    dataset_out = OUT_TEST_DIR / "compiled" / test_name
    if dataset_out.exists():
        shutil.rmtree(dataset_out.resolve())

    loudness_feature = features.Loudness()
    writer = LMDBWriter(get_dataset(dataset), dataset_out, features=[loudness_feature], waveform=waveform)
    writer.build()
    assert read_metadata(dataset_out)['waveform'] == waveform
    n_chunks = read_metadata(dataset_out)['n_chunks']

    # add data : waveform must follow the original dataset setting
    LMDBWriter.update(dataset_out, data=[get_dataset(additional_dataset)], check=False, overwrite=True)
    assert read_metadata(dataset_out)['n_chunks'] > n_chunks
    check_dataset_loading(dataset_out, waveform, features=[loudness_feature.feature_name], test_k=test_k)
