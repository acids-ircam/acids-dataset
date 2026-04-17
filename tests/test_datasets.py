import os, sys
import gin
import random
import lmdb
import shutil
import torch
import torchaudio
import pytest
import gin

from . import OUT_TEST_DIR, test_name
from pathlib import Path
from acids_dataset.writers import audio_paths_from_dir, LMDBWriter, read_metadata
from acids_dataset.parsers import raw_parser as raw
from acids_dataset.datasets import AudioDataset
from acids_dataset.utils import loudness, set_gin_constant
from acids_dataset import transforms
from acids_dataset import get_fragment_class, features, preprocess_dataset
from .datasets import get_available_datasets, get_dataset, get_available_datasets_with_filters


@pytest.mark.parametrize("dataset,filters", get_available_datasets_with_filters())
def test_parse_dataset_files(dataset, filters, test_name):
    dataset_path = get_dataset(dataset)
    flt, ex = filters
    valid_files = audio_paths_from_dir(dataset_path, flt=flt, exclude=ex)
    with open(Path(OUT_TEST_DIR) / f"{test_name}.txt", "w+") as f:
        f.write(f"filters: {flt}\n")
        f.write(f"exclude: {ex}\n")
        f.write("\n".join(valid_files))

# @pytest.mark.parametrize("dataset,filters", get_available_datasets_with_filters())
@pytest.mark.parametrize("loudness_threshold", [None, -70, -10])
@pytest.mark.parametrize("hop_length,overlap", [(0.5, None), (None, 0.8), (4096, None), (None, 4096)])
@pytest.mark.parametrize("chunk_length", [1., 8192])
@pytest.mark.parametrize("pad_mode", list(raw.PadMode.__members__.keys()))
@pytest.mark.parametrize("import_backend", list(raw.ImportBackend.__members__.keys()))
@pytest.mark.parametrize("parser", [raw.RawParser])
@pytest.mark.parametrize("dataset", get_available_datasets())
def test_raw_parser(dataset, test_name, parser, import_backend, pad_mode, chunk_length, hop_length, overlap, loudness_threshold):
    dataset_path = get_dataset(dataset)
    valid_files = audio_paths_from_dir(dataset_path)
    file_exceptions = []
    file_stats = {}
    for path in valid_files:
        current_parser = parser(
            path, 
            chunk_length=chunk_length, 
            hop_length=hop_length, 
            overlap=overlap, 
            sr = 44100,
            channels = 1,
            pad_mode=pad_mode, 
            import_backend=import_backend, 
            loudness_threshold=loudness_threshold            
        )
        for obj in current_parser:
            try:
                data = obj()
                assert len(list(filter(lambda x, l = current_parser.chunk_length_smp: x['audio'].shape[-1] != l, data))) == 0
                if loudness_threshold is not None:
                    assert len(list(filter(lambda x, t = loudness_threshold, sr = current_parser.sr: loudness(torch.Tensor(x['audio']), sr) < t, data))) == 0
                file_stats[path] = {"n_chunks": len(data)}
            except raw.FileNotReadException as e:
                file_exceptions.append(e)

    with open(Path(OUT_TEST_DIR) / f"{test_name}.txt", "w+") as f:
        # f.write(f"--failed files: \n{'\n'.join(map(str, file_exceptions))}")
        f.write("--failed files: \n")
        f.write('\n'.join(map(str, file_exceptions)))
        f.write("\n--filewise information :\n")
        for p in valid_files:
            p_name = Path(p).relative_to(dataset_path)
            stat = file_stats.get(p, "MISSING")
            f.write(f"{p_name}\t{stat}\n")
            
@pytest.mark.parametrize('config', ['rave.gin'])
@pytest.mark.parametrize("dataset", get_available_datasets())
def test_build_dataset(config, dataset, test_name, test_k = 1):
    # test writing
    set_gin_constant('SAMPLE_RATE', 44100)
    set_gin_constant('CHANNELS', 1)
    gin.parse_config_file(config)
    dataset_path = get_dataset(dataset)
    dataset_out = OUT_TEST_DIR / "compiled" / test_name
    if dataset_out.exists():
        shutil.rmtree(dataset_out.resolve())
    writer = LMDBWriter(dataset_path, dataset_out)
    writer.build()

    # test loading
    env = lmdb.open(str(dataset_out), lock=False, readonly=True)
    fragment_class = get_fragment_class(read_metadata(dataset_out)['fragment_class'])
    with env.begin() as txn:
        dataset_keys = list(txn.cursor().iternext(values=False))
        # pick a random item
        random_keys = random.choices(dataset_keys, k=test_k)
        for key in random_keys:
            ae = fragment_class(txn.get(key))
            audio = ae.get_audio("waveform")



@pytest.mark.parametrize('config', ['default.gin'])
def test_slakh_dataset(config, test_name, test_k=2):
    # test writing
    set_gin_constant('CHUNK_LENGTH', 88200)
    set_gin_constant('HOP_LENGTH', 88200)
    set_gin_constant('SAMPLE_RATE', 44100)
    set_gin_constant('CHANNELS', 1)
    gin.parse_config_file(config)
    dataset_path = Path(__file__).parent / "datasets" / "slakh_like"
    dataset_out = OUT_TEST_DIR / "compiled" / test_name
    if dataset_out.exists():
        shutil.rmtree(dataset_out.resolve())
    writer = LMDBWriter(dataset_path, dataset_out, filters=("**/stems/*"), features=[features.AfterMIDI(), features.Loudness()])
    writer.build()

    # test loading
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
            loudness = ae.get_array("loudness")



@pytest.mark.parametrize("dataset", get_available_datasets())
@pytest.mark.parametrize("channels", [None, 1, 2, 8])
@pytest.mark.parametrize('config', ['default.gin'])
@pytest.mark.parametrize("keep_gin_constants", [True, False])
def test_preprocessing_fn(dataset, config, channels, keep_gin_constants, test_name):
    gin.clear_config(clear_constants=True)
    dataset_path = Path(__file__).parent / "datasets" / test_name
    os.makedirs(dataset_path, exist_ok=True)
    dataset = preprocess_dataset(
        path = get_dataset(dataset), 
        out = dataset_path, 
        config=config,
        chunk_length=10000, 
        hop_length=2000,
        sample_rate=48000,
        channels=channels,
        keep_gin_constants=keep_gin_constants, 
        force=True
    )

    assert gin.config_str() == ""
    assert gin.operative_config_str() == ""
    constants = dict(gin.config._CONSTANTS.items())
    constants.pop('gin.REQUIRED')
    if keep_gin_constants: 
        assert len(constants) > 0
    else:
        assert len(constants) == 0

    



    


        

