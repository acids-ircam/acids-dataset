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
from acids_dataset.presets import parse_preset, quiet_gin_search

def import_flags():
    flags.DEFINE_multi_string('path', None, help="dataset path", required=True)
    flags.DEFINE_string('out', None, help="parsed dataset location")
    flags.DEFINE_string('config', "default", help="dataset preset (name or path to a .gin file)")
    flags.DEFINE_alias('preset', 'config')
    flags.DEFINE_string('device', "cpu", help="device for feature computation")
    flags.DEFINE_multi_string('feature', [], help="config files")
    flags.DEFINE_multi_string('filter', [], help="wildcard to filter target files")
    flags.DEFINE_multi_string('exclude', [], help="wildcard to exclude target files")
    flags.DEFINE_multi_string('meta_regexp', [], help="additional regexp for metadata parsing")
    flags.DEFINE_multi_string('override', [], help="additional overridings for configs")
    flags.DEFINE_integer('sample_rate', None, help="sample rate (overrides preset)")
    flags.DEFINE_integer('channels', None, help="number of audio channels (overrides preset)")
    flags.DEFINE_boolean('check', True, help="has interactive mode for data checking.")
    flags.DEFINE_boolean('force', False, help="force dataset preprocessing if folder already exists")
    flags.DEFINE_float('max_db_size', 100, help="maximum database size")
    flags.DEFINE_integer('chunk_length', None, help="number of samples per chunk (overrides preset)")
    flags.DEFINE_boolean('waveform', True, help="store waveform in the database (--nowaveform to only store features)")
    flags.DEFINE_boolean('compact', False, help="create compact version of lmdb database")
    flags.DEFINE_string('log', None, help="log file")

def _files_parser(files: List[str]):
    """dir_parser of writers that gives the files of a list that are in a folder."""
    def dir_parser(path, valid_exts=None, flt=None, exclude=None):
        root = Path(path).resolve()
        return [f for f in files if Path(f).is_relative_to(root)]
    return dir_parser


def get_default_output_path(path):
    return (Path(".") / f"{path.stem}_preprocessed").resolve().absolute()


def preprocess_dataset(
    path: List[str], 
    out: str = None, 
    config: str = 'default', 
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
    keep_gin_constants: bool = False,
    preset: str | None = None,
    files: List[str] | None = None,
    metadata: dict | None = None
    ):
    """pre-processes audio folders with a preset (see acids_dataset.presets). Given values (chunk_length, sample_rate, 
    channels, ...) override the ones of the preset ; preset can also be given with the config keyword. If files is given, 
    only these audio files (absolute paths in the given folders) are parsed, instead of the files found in folders ; 
    metadata is recorded in the metadata of the dataset."""
    preset = preset or config
    with GinEnv(keep_constants=keep_gin_constants):
        path = list(map(Path, checklist(path)))
        import_database_configs(*path)
        # parse features
        features = features or []
        out = out or get_default_output_path(path)

        if chunk_length and not hop_length:
            hop_length = chunk_length // 2 if isinstance(chunk_length, int) else chunk_length / 2
        parse_preset(preset, overrides=override, set_constants=True, values={
            'CHUNK_LENGTH': chunk_length, 
            'HOP_LENGTH': hop_length, 
            'SAMPLE_RATE': sample_rate, 
            'CHANNELS': channels, 
            'DEVICE': device
        })

        # append additional features
        operative_features = parse_features()
        operative_features = append_meta_regexp(operative_features, meta_regexp=meta_regexp)
        for i, f in enumerate(features):
            if isinstance(f, str):
                with quiet_gin_search():
                    operative_features.extend(feature_from_gin_config(f))
            elif isinstance(f, AcidsDatasetFeature):
                operative_features.append(f)

        gin.finalize() 
        writer_class = get_writer_class(filters=flt, exclude=exclude)
        writer_kwargs = {}
        if files is not None:
            writer_kwargs['dir_parser'] = _files_parser([str(Path(f).resolve()) for f in files])
        writer = writer_class(path, 
                            out, 
                            features=operative_features, 
                            check=check, 
                            force=force, 
                            waveform=waveform, 
                            max_db_size=max_db_size, 
                            log=log, 
                            **writer_kwargs) 
        writer.metadata['preset'] = str(preset)
        writer.metadata.update(metadata or {})
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
