"""Presets are gin configuration files describing how a dataset is parsed (sampling rate, channels, chunking,
features...). They are shared by pre-processing (preprocess_dataset) and online datasets (OnlineAudioDataset),
such that a given preset gives the same data in both modes.

A preset is a top-level .gin file of the package configs (acids_dataset/configs) or of an `ad_configs` folder in
the current directory (that takes precedence), or a path to a .gin file. Presets define the following values,
that are used by the bindings of default.gin (that presets usually include) :

    SAMPLE_RATE   target sampling rate
    CHANNELS      target number of channels (None to keep original channels)
    CHUNK_LENGTH  chunk length, in samples (int) or seconds (float)
    HOP_LENGTH    hop length between pre-processed chunks (None for chunk_length)
    DEVICE        device for feature computation

Values explicitly given by the user override the ones of the preset.
"""
import os
import logging
import contextlib
import gin
from pathlib import Path
from typing import Dict, List, Any
from . import ACIDS_DATASET_CONFIG_PATH
from .utils import GinEnv, set_gin_constant, checklist

PRESET_VALUES = ["SAMPLE_RATE", "CHANNELS", "CHUNK_LENGTH", "HOP_LENGTH", "DEVICE"]


def preset_paths() -> List[Path]:
    """folders where presets are looked for, by order of precedence."""
    # resolved at call time, as the current directory may change
    return [Path(os.getcwd()) / "ad_configs", ACIDS_DATASET_CONFIG_PATH]


def list_presets() -> Dict[str, Path]:
    """returns available presets as {name: path}."""
    presets = {}
    for folder in reversed(preset_paths()):
        if not folder.is_dir(): continue
        for f in sorted(folder.iterdir()):
            if f.suffix == ".gin" and f.is_file():
                presets[f.stem] = f
    return dict(sorted(presets.items()))


def get_preset_path(preset: str | Path) -> Path:
    """returns the path of a preset, given as a name (e.g. "rave") or as a path to a .gin file."""
    preset_path = Path(preset)
    if preset_path.suffix == ".gin" and preset_path.is_file():
        return preset_path.resolve()
    presets = list_presets()
    name = preset_path.stem if preset_path.suffix == ".gin" else str(preset)
    if name not in presets:
        raise FileNotFoundError("preset %s not found ; available presets : %s"%(preset, list(presets.keys())))
    return presets[name]


class _GinSearchPathFilter(logging.Filter):
    # gin logs each search path that does not contain a config file, although the file is found in another 
    # one (gin raises an error if the file is not found at all)
    def filter(self, record):
        message = record.getMessage()
        return not (message.startswith("Path not found:") or message.startswith("system_path_file_exists:"))


@contextlib.contextmanager
def quiet_gin_search():
    """hides gin logs for config files lookups in search paths."""
    log_filter = _GinSearchPathFilter()
    # filters of loggers do not apply to records of child loggers : filter handlers too
    filtered = [logging.getLogger()] + list(logging.getLogger().handlers)
    for f in filtered: f.addFilter(log_filter)
    try:
        yield
    finally:
        for f in filtered: f.removeFilter(log_filter)


def _value_binding(name: str, value: Any) -> str:
    return "%s = %r"%(name, value)


def parse_preset(preset: str | Path | None = None, overrides: List[str] = [], values: Dict[str, Any] | None = None, set_constants: bool = False) -> None:
    """parses a preset in the current gin config. Values (e.g. {"SAMPLE_RATE": 22050}) that are not None
    override the ones of the preset, and overrides are additional gin bindings."""
    values = {k: v for k, v in (values or {}).items() if v is not None}
    for p in preset_paths():
        gin.add_config_file_search_path(str(p))
    # gin constants take precedence over the values of presets : constants left by previous configurations (e.g. 
    # set_gin_constant, or preprocess_dataset with keep_gin_constants) are removed, such that presets give their 
    # values. Callers parse presets in a GinEnv, that restores previous constants at exit.
    for name in PRESET_VALUES:
        if name in gin.config._CONSTANTS:
            gin.config._CONSTANTS.pop(name)
    if set_constants:
        # retro-compatibility : configs may refer to values defined as gin constants
        for k, v in values.items():
            set_gin_constant(k, v)
    configs = [] if preset is None else [str(get_preset_path(preset))]
    bindings = [_value_binding(k, v) for k, v in values.items()] + checklist(overrides)
    with quiet_gin_search():
        gin.parse_config_files_and_bindings(configs, bindings, finalize_config=False)


@contextlib.contextmanager
def preset_env(preset: str | Path | None = None, overrides: List[str] = [], values: Dict[str, Any] | None = None, keep_constants: bool = False, set_constants: bool = False):
    """context in which a preset is parsed ; the previous gin config is restored at exit."""
    with GinEnv(keep_constants=keep_constants):
        parse_preset(preset, overrides=overrides, values=values, set_constants=set_constants)
        yield


def query_preset_value(name: str, default: Any = None) -> Any:
    """returns the value of a preset value in the current gin config."""
    try:
        return gin.query_parameter("%" + name)
    except ValueError:
        return default


def get_preset_values(preset: str | Path, overrides: List[str] = []) -> Dict[str, Any]:
    """returns the values (SAMPLE_RATE, CHANNELS...) defined by a preset."""
    with preset_env(preset, overrides=overrides):
        return {k: query_preset_value(k) for k in PRESET_VALUES}



def import_flags():
    from absl import flags
    flags.DEFINE_string('preset', None, help="show the content of a given preset")


def main(argv):
    """lists available presets with their values, or prints a given preset."""
    from absl import flags
    FLAGS = flags.FLAGS
    if FLAGS.preset is not None:
        preset_path = get_preset_path(FLAGS.preset)
        print("# %s\n"%preset_path)
        print(preset_path.read_text())
        return
    for name, path in list_presets().items():
        try:
            values = get_preset_values(name)
            values = ", ".join("%s=%s"%(k, v) for k, v in values.items())
        except Exception as e:
            values = "could not parse preset (%s)"%e
        print("%s\t%s\n\t%s"%(name, path, values))

__all__ = ['quiet_gin_search', 'PRESET_VALUES', 'preset_paths', 'list_presets', 'get_preset_path', 'parse_preset', 'preset_env', 'query_preset_value', 'get_preset_values']
