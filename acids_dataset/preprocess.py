import os
import re
import torch
import gin
import logging
from pathlib import Path
from typing import List
from absl import flags, app

import sys
import sys; sys.path.append(str(Path(__file__).parent.parent))
import acids_dataset as ad
from acids_dataset import import_database_configs, get_feature_config_paths
from acids_dataset.fragments.acids import AcidsFragment
# for retro-compatibility
AcidsFragment.force_array_reshape = False
from acids_dataset.writers import get_writer_class
from acids_dataset.features import AcidsDatasetFeature, append_meta_regexp
from acids_dataset.utils import feature_from_gin_config, parse_features, checklist, set_gin_constant, GinEnv

def import_flags():
    flags.DEFINE_multi_string('path', None, help="dataset path", required=True)
    flags.DEFINE_string('out', None, help="parsed dataset location")
    flags.DEFINE_string('config', "default.gin", help="dataset config")
    flags.DEFINE_string('device', "cpu", help="device for feature computation")
    flags.DEFINE_multi_string('feature', [], help="config files")
    flags.DEFINE_multi_string('filter', [], help="wildcard to filter target files")
    flags.DEFINE_multi_string('exclude', [], help="wildcard to exclude target files")
    flags.DEFINE_multi_string('meta_regexp', [], help="additional regexp for metadata parsing")
    flags.DEFINE_multi_string('override', [], help="additional overridings for configs")
    flags.DEFINE_integer('sample_rate', 44100, help="sample rate")
    flags.DEFINE_integer('channels', None, help="number of audio channels")
    flags.DEFINE_boolean('check', True, help="has interactive mode for data checking.")
    flags.DEFINE_boolean('force', False, help="force dataset preprocessing if folder already exists")
    flags.DEFINE_float('max_db_size', 100, help="maximum database size")
    flags.DEFINE_integer('chunk_length', None, help="number of samples per chunk")
    flags.DEFINE_boolean('waveform', True, help="no waveform parsing")
    flags.DEFINE_boolean('compact', False, help="create compact version of lmdb database")
    flags.DEFINE_string('log', None, help="log file")

def get_default_output_path(path):
    return (Path(".") / f"{path.stem}_preprocessed").resolve().absolute()


def preprocess_dataset(
    path: List[str], 
    out: str = None, 
    config: str = 'default.gin', 
    features: List[str | AcidsDatasetFeature] | None = None,
    check=False, 
    chunk_length: int = None,
    hop_length: int = None, 
    sample_rate = None,
    channels = None,
    flt = [],
    exclude = [],
    meta_regexp = [], 
    force: bool = False, 
    waveform: bool = True, 
    override = [],
    device: str | None = None, 
    max_db_size: int = 100, 
    compact: bool = False, 
    log: str | None = None, 
    keep_gin_constants: bool = False
    ):
    with GinEnv(keep_constants=keep_gin_constants):
        path = list(map(Path, checklist(path)))
        import_database_configs(*path)
        # parse features
        features = features or []
        out = out or get_default_output_path(path)

        if chunk_length:
            set_gin_constant('CHUNK_LENGTH', chunk_length)
            set_gin_constant('HOP_LENGTH', hop_length or chunk_length // 2)
        if sample_rate: set_gin_constant('SAMPLE_RATE', sample_rate)
        set_gin_constant('CHANNELS', channels)
        set_gin_constant('DEVICE', device or "cpu")

        if os.path.splitext(config)[-1] != ".gin":
            config += ".gin"
        gin.parse_config_files_and_bindings([config], override, finalize_config=False)

        # append additional features
        operative_features = parse_features()
        operative_features = append_meta_regexp(operative_features, meta_regexp=meta_regexp)
        for i, f in enumerate(features):
            if isinstance(f, str):
                operative_features.extend(feature_from_gin_config(f))
            elif isinstance(f, AcidsDatasetFeature):
                operative_features.append(f)

        gin.finalize() 
        writer_class = get_writer_class(filters=flt, exclude=exclude)
        writer = writer_class(path, 
                            out, 
                            features=operative_features, 
                            check=check, 
                            force=force, 
                            waveform=waveform, 
                            max_db_size=max_db_size, 
                            log=log) 
        writer.build(compact=compact)

def main(argv):

    FLAGS = flags.FLAGS

    preprocess_dataset(
        FLAGS.path, 
        out = FLAGS.out, 
        config = FLAGS.config, 
        device = FLAGS.device, 
        features = FLAGS.feature,
        check = FLAGS.check, 
        chunk_length = FLAGS.chunk_length, 
        sample_rate=FLAGS.sample_rate, 
        channels = FLAGS.channels, 
        flt=FLAGS.filter, 
        exclude=FLAGS.exclude,
        meta_regexp=FLAGS.meta_regexp,
        force=FLAGS.force,
        waveform=FLAGS.waveform, 
        override=FLAGS.override,
        max_db_size = FLAGS.max_db_size, 
        compact=FLAGS.compact,
        log=FLAGS.log
    )

if __name__ == "__main__":
    import_flags() 
    app.run(main)

__all__ = ['preprocess_dataset']
