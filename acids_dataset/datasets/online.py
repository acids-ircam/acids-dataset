import os
import json
import copy
import math
import bisect
import random
import statistics
import gin
import numpy as np
import torch, torchaudio
from absl import logging
from pathlib import Path
from types import EllipsisType
from typing import List, Dict, Any, Iterable, Literal
from ..utils import checklist, audio_info, load_audio_range, pad, PadMode, parse_features, feature_from_gin_config
from ..writers.utils import audio_paths_from_dir
from ..features import AcidsDatasetFeature, append_meta_regexp
from .. import get_fragment_class
from .base import AudioDataset, TransformType, chunk_length_to_samples
from .utils import is_multi_index, _outs_from_pattern, _check_channels
from .batch import parse_output_pattern, transforms_per_entry, augmented_audio_entry, entry_name
from .baked import BakedFeature, BAKED_FOLDER, BAKED_VERSION, baked_file_path, read_baked_file, read_baked_metadata, write_baked_file, probe_frame_rate, hop_hint, to_numpy


BatchByType = Literal["files", "dataset", "auto"]


def _entry_from_fragment(fragment, source: str, index=None):
    """output of a single entry of an output pattern (e.g. "waveform[0]", "loudness")."""
    return _outs_from_pattern(fragment, "(%s)"%source, index=index)[0]


@gin.configurable(module="datasets")
class OnlineAudioDataset(AudioDataset):
    """AudioDataset that parses audio files on the fly instead of reading a pre-processed database.

    At initialization, target audio files are scanned for header information (length, sampling rate,
    channels, codec) and path-based metadata (meta_regexp). This information is cached in an index
    file (".ad" by default, in each dataset folder) so that further instantiations only re-scan
    new or modified files. Features are computed dynamically when audio is loaded.

    Items are chunks of chunk_length samples, whose distribution is set by batch_by:
        - "files": one item per file, whose position is sampled randomly in the file. Files shorter than 
            chunk_length are padded with pad_mode.
        - "dataset": each file is split in max(1, round(duration / chunk_length)) slots, such that the 
            dataset length is approximately total_duration / chunk_length. Each item is sampled randomly 
            within its slot, such that every position of the dataset may be sampled while items 
            never overlap two files.
        - "auto": "dataset" if median file duration is greater than auto_ratio * chunk_length 
            (a few long files), "files" otherwise (a lot of short files).
    """
    index_name = ".ad"
    index_version = 1
    batch_by_modes = ["files", "dataset", "auto"]
    _preset_values = {'sr': 'SAMPLE_RATE', 'channels': 'CHANNELS', 'chunk_length': 'CHUNK_LENGTH'}
    # context (in original samples) loaded around chunks before resampling
    _resampling_context = 256

    @classmethod
    def _init_from_preset(cls, preset, preset_overrides, *args, **kwargs):
        from ..presets import preset_env
        with preset_env(preset, overrides=preset_overrides, values=cls._preset_values_from_kwargs(kwargs)):
            # features of the preset are completed with given features, as in preprocess_dataset
            kwargs['features'] = parse_features() + cls._parse_feature_list(kwargs.get('features'))
            return type.__call__(cls, *args, **kwargs)

    @staticmethod
    def _feature_items(features) -> List[tuple]:
        """returns features as (name, feature) items ; names are given by dict keys, and None otherwise."""
        if isinstance(features, dict):
            return list(features.items())
        return [(None, f) for f in checklist(features or [])]

    @classmethod
    def _parse_feature_list(cls, features) -> List[AcidsDatasetFeature]:
        """features can be given as a list or a {name: feature} dict of feature objects or feature config names 
        (e.g. "loudness"), that are parsed in the current gin config. Dict keys set feature names."""
        from ..presets import quiet_gin_search
        parsed_features = []
        with quiet_gin_search():
            for name, f in cls._feature_items(features):
                parsed = feature_from_gin_config(f) if isinstance(f, (str, Path)) else [f]
                if name is not None:
                    if len(parsed) != 1:
                        raise ValueError("feature %s gives several features, and cannot be renamed"%name)
                    parsed[0].feature_name = name
                parsed_features.extend(parsed)
        return parsed_features

    def __init__(self,
                 path: str | Path | List[str | Path],
                 transforms: TransformType = None,
                 output_pattern: str = 'waveform',
                 *,
                 output_type: str = "torch",
                 channels: int | None = 1,
                 sr: int | None = None,
                 chunk_length: int | float | None = None,
                 batch_by: BatchByType = "auto",
                 pad_mode: PadMode | str = PadMode.ZERO,
                 auto_ratio: float = 2.,
                 loudness_threshold: float | None = None,
                 max_retries: int = 10,
                 index_dir: str | Path | None = None,
                 partition_dir: str | Path | None = None,
                 features: List[AcidsDatasetFeature] | None = None,
                 meta_regexp: List[str] = [],
                 filters: List[str] = [],
                 exclude: List[str] = [],
                 valid_exts: List[str] | None = None,
                 bformat: str = "float32",
                 fragment_class: str = "AcidsFragment",
                 rescan: bool = False,
                 write_index: bool = True,
                 max_samples: int | None = None,
                 features_after_transforms: bool = True,
                 **kwargs) -> None:
        """
        Args:
            path (str | Path | List[str | Path]): dataset folder(s).
            sr (int | None, optional): target sampling rate, None to keep original sampling rates. Defaults to None.
            channels (int | None, optional): target number of channels, None to keep original channels. Defaults to 1.
            chunk_length (int | float | None, optional): item length, in samples (int) or seconds (float). Can also be 
                given to configure / get_loader. Defaults to None.
            batch_by (str, optional): item sampling strategy ("files", "dataset", or "auto"). Defaults to "auto".
            pad_mode (PadMode | str, optional): padding mode for chunks exceeding files. Defaults to PadMode.ZERO.
            auto_ratio (float, optional): median duration / chunk_length ratio above which "auto" selects "dataset". Defaults to 2.
            loudness_threshold (float | None, optional): chunks with a loudness (LUFS) below this threshold (e.g. silence) are 
                drawn again at a random position of their file, up to max_retries times. Defaults to None.
            max_retries (int, optional): maximum number of draws of chunks below loudness_threshold. Defaults to 10.
            index_dir (str | Path | None, optional): folder where index files are written. If None, index is written
                as a ".ad" file in each dataset folder ; otherwise, as "<folder_name>.ad" in index_dir. Defaults to None.
            partition_dir (str | Path | None, optional): folder where partitions are written. If None, partitions are 
                written in "partitions" of index_dir if given, or in ".ad_partitions" of the first dataset folder. Defaults to None.
            features (List[AcidsDatasetFeature | str] | Dict[str, AcidsDatasetFeature | str] | None, optional): features computed 
                dynamically on loaded audio, given as objects or feature config names (e.g. "loudness") ; dict keys set 
                feature names. Defaults to None.
            features_after_transforms (bool, optional): compute features on augmented audio, i.e. on the waveform output 
                after its transforms (True), or on loaded audio, before transforms (False). Can also be given to configure 
                and get_loader. Defaults to True.
            meta_regexp (List[str], optional): glob-like patterns for path-based metadata (e.g. "**/{{instrument}}_*.wav"). Defaults to [].
            filters (List[str], optional): glob-like patterns to filter audio paths. Defaults to [].
            exclude (List[str], optional): glob-like patterns to exclude audio paths. Defaults to [].
            valid_exts (List[str] | None, optional): valid audio extensions, None for default ones. Defaults to None.
            bformat (str, optional): audio format of loaded fragments. Defaults to "float32".
            rescan (bool, optional): ignore existing index files and re-scan every file. Defaults to False.
            write_index (bool, optional): write index files after scanning. Defaults to True.
        """
        self._paths = [Path(p).resolve() for p in checklist(path)]
        for p in self._paths:
            if not p.exists(): raise FileNotFoundError("%s folder does not exist."%p)
        # audio paths and index files are named after dataset folders
        folder_names = [p.name for p in self._paths]
        if len(set(folder_names)) != len(folder_names):
            raise ValueError("dataset folders must have distinct names, got %s"%self._paths)
        self._init_common(transforms=transforms, output_pattern=output_pattern, output_type=output_type, 
                          channels=channels, max_samples=max_samples, features_after_transforms=features_after_transforms)
        self._index_dir = None if index_dir is None else Path(index_dir).resolve()
        self._partition_dir = None if partition_dir is None else Path(partition_dir).resolve()
        self._filters = checklist(filters)
        self._exclude = checklist(exclude)
        self._valid_exts = valid_exts
        self._meta_regexp = checklist(meta_regexp)
        self._regexp_features = append_meta_regexp([], self._meta_regexp)
        from .simple import PreprocessedAudioDataset
        self._check_kwargs(kwargs, PreprocessedAudioDataset)
        if any(isinstance(f, (str, Path)) for _, f in self._feature_items(features)):
            # feature configs are parsed with the default preset, with dataset values
            from ..presets import preset_env
            with preset_env("default", values={'SAMPLE_RATE': sr, 'CHANNELS': channels, 'CHUNK_LENGTH': chunk_length}):
                features = self._parse_feature_list(features)
        else:
            features = self._parse_feature_list(features)
        self._features = list(features)
        self._sr = sr
        self._bformat = bformat
        self._fragment_class = get_fragment_class(fragment_class)
        self._db_path = self._paths[0]
        self._files = []
        self._invalid_files = []
        for p in self._paths:
            self._scan(p, rescan=rescan, write_index=write_index)
        if len(self._files) == 0:
            raise RuntimeError("no valid audio files found in %s"%self.path)
        self._auto_ratio = auto_ratio
        self._loudness_threshold = loudness_threshold
        self._max_retries = max_retries
        self._chunk_length = None
        self._batch_by = None
        self._pad_mode = None
        self._set_loading_params(chunk_length=chunk_length, batch_by=batch_by, pad_mode=pad_mode)
        torch.utils.data.Dataset.__init__(self)

    # index handling

    @classmethod
    def index_path_for(cls, dataset_path: str | Path, index_dir: str | Path | None = None) -> Path:
        """returns the index file location for a given dataset folder, without instantiating the dataset."""
        dataset_path = Path(dataset_path).resolve()
        if index_dir is None:
            return dataset_path / cls.index_name
        return Path(index_dir).resolve() / f"{dataset_path.name}{cls.index_name}"

    def index_path(self, dataset_path: str | Path) -> Path:
        """returns the index file location for a given dataset folder."""
        return self.index_path_for(dataset_path, self._index_dir)

    def _read_index(self, dataset_path: Path) -> Dict[str, Any]:
        index_path = self.index_path(dataset_path)
        if not index_path.exists():
            return {}
        try:
            with open(index_path, "r") as f:
                index = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logging.warning("could not read index %s (%s) ; re-scanning"%(index_path, e))
            return {}
        if index.get('version') != self.index_version:
            return {}
        return index

    def _write_index(self, dataset_path: Path, index: Dict[str, Any]) -> None:
        index_path = self.index_path(dataset_path)
        os.makedirs(index_path.parent, exist_ok=True)
        tmp_path = index_path.with_name(index_path.name + ".tmp")
        with open(tmp_path, "w+") as f:
            json.dump(index, f, indent=1)
        os.replace(tmp_path, index_path)

    @staticmethod
    def _file_stamp(file_path: Path) -> Dict[str, Any]:
        stat = os.stat(file_path)
        return {'size': stat.st_size, 'mtime': stat.st_mtime}

    def _parse_meta_regexp(self, audio_path: str) -> Dict[str, Any]:
        return {f.feature_name: f._extract_id_from_path(audio_path) for f in self._regexp_features}

    def _scan(self, dataset_path: Path, rescan: bool = False, write_index: bool = True) -> None:
        """scans audio files of dataset_path, using the index as a cache for unmodified files."""
        index = {} if rescan else self._read_index(dataset_path)
        cached_files = index.get('files', {})
        cached_invalid = index.get('invalid', {})
        audio_files = audio_paths_from_dir(dataset_path, valid_exts=self._valid_exts, flt=self._filters, exclude=self._exclude)
        n_scanned = 0
        for audio_file in sorted(audio_files):
            audio_file = Path(audio_file)
            rel_path = str(audio_file.relative_to(dataset_path))
            stamp = self._file_stamp(audio_file)
            entry = cached_files.get(rel_path)
            if entry is None or any(entry.get(k) != v for k, v in stamp.items()):
                if cached_invalid.get(rel_path) == stamp:
                    continue
                try:
                    entry = {**audio_info(audio_file), **stamp}
                except Exception as e:
                    logging.warning("could not read file %s : %s"%(audio_file, e))
                    cached_files.pop(rel_path, None)
                    cached_invalid[rel_path] = stamp
                    continue
                cached_invalid.pop(rel_path, None)
                n_scanned += 1
            # path-based metadata use the same convention as LMDBWriter (path relative to dataset parent)
            entry['metadata'] = self._parse_meta_regexp(str(Path(dataset_path.name) / rel_path))
            cached_files[rel_path] = entry
            self._files.append({'path': str(audio_file), 'audio_path': str(Path(dataset_path.name) / rel_path), **entry})
        self._invalid_files.extend(str(dataset_path / p) for p in cached_invalid)

        # remove deleted files from cache
        for rel_path in list(cached_files.keys()):
            if not (dataset_path / rel_path).exists(): del cached_files[rel_path]
        for rel_path in list(cached_invalid.keys()):
            if not (dataset_path / rel_path).exists(): del cached_invalid[rel_path]

        if write_index:
            self._write_index(dataset_path, {
                'version': self.index_version,
                'dataset_path': str(dataset_path),
                'meta_regexp': self._meta_regexp,
                'files': cached_files,
                'invalid': cached_invalid
            })
        logging.info("%s : %d files (%d scanned, %d invalid)"%(dataset_path, len(audio_files), n_scanned, len(cached_invalid)))

    # properties

    @property
    def path(self):
        return [str(p) for p in self._paths]

    @property
    def files(self) -> List[Dict[str, Any]]:
        """list of every scanned audio file, with header information and path-based metadata. File indices 
        (used in get_fragment, chunk_position, partitions...) refer to this list."""
        return copy.deepcopy(self._files)

    @property
    def file_indices(self) -> List[int]:
        """indices of files used by the dataset (all files, or a subset for partitions)."""
        return list(range(len(self._files))) if self._subindices is None else self._subindices

    @property
    def invalid_files(self) -> List[str]:
        return list(self._invalid_files)

    @property
    def features(self) -> List[AcidsDatasetFeature]:
        return list(self._features)

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            'n_files': len(self.file_indices),
            'n_seconds': sum(self._files[i]['duration'] for i in self.file_indices),
            'sr': self.sr,
            'channels': self._n_channels,
            'chunk_length': self.chunk_length,
            'batch_by': self.batch_by,
            'loudness_threshold': self.loudness_threshold,
            'pad_mode': self._pad_mode.name.lower(),
            'features': {f.feature_name: str(f) for f in self._features},
            'meta_regexp': self._meta_regexp,
            'filters': self._filters,
            'exclude': self._exclude,
        }

    @property
    def loader(self):
        raise NotImplementedError("OnlineAudioDataset has no database loader.")

    @property
    def feature_hash(self):
        raise NotImplementedError()

    @property
    def keys(self):
        raise NotImplementedError()

    # loading parameters

    @property
    def sr(self) -> int | None:
        """sampling rate of dataset items ; None if not set and files have different sampling rates."""
        if self._sr is not None:
            return self._sr
        srs = set(f['sr'] for f in self._files)
        return srs.pop() if len(srs) == 1 else None

    @property
    def chunk_length(self) -> int | None:
        """chunk length of dataset items, in samples."""
        if self._chunk_length is None:
            return None
        if isinstance(self._chunk_length, int):
            return self._chunk_length
        if self.sr is None:
            raise ValueError("files have different sampling rates ; please provide sr to dataset")
        return chunk_length_to_samples(self._chunk_length, self.sr)

    @property
    def batch_by(self) -> str | None:
        """resolved item sampling strategy ("files" or "dataset") ; "files" (one whole file per item) if chunk_length is None."""
        return self._batch_by

    @property
    def pad_mode(self) -> PadMode:
        return self._pad_mode

    def _resolve_batch_by(self, batch_by: str) -> str:
        if batch_by not in self.batch_by_modes:
            raise ValueError("batch_by must be in %s, got %s"%(self.batch_by_modes, batch_by))
        if batch_by != "auto":
            return batch_by
        # resolved on every file, such that partitions share the same mode
        median_duration = statistics.median(f['duration'] for f in self._files)
        return "dataset" if median_duration >= self._auto_ratio * self.chunk_length / self.sr else "files"

    def _set_loading_params(self, chunk_length=None, batch_by=None, pad_mode=None) -> None:
        if pad_mode is not None:
            if isinstance(pad_mode, str): pad_mode = getattr(PadMode, pad_mode.upper())
            if pad_mode == PadMode.DISCARD:
                raise ValueError("PadMode.DISCARD is not available for OnlineAudioDataset")
            self._pad_mode = pad_mode
        if chunk_length is not None:
            if chunk_length <= 0: raise ValueError("chunk_length must be positive")
            self._chunk_length = chunk_length
        if batch_by is not None:
            self._batch_by_arg = batch_by
        if self._chunk_length is None:
            # without chunk_length, items are whole files
            self._batch_by = "files"
            self._build_chunk_map()
            return
        if self.sr is None:
            raise ValueError("files have different sampling rates ; please provide sr to dataset")
        self._batch_by = self._resolve_batch_by(self._batch_by_arg)
        self._build_chunk_map()

    def _n_chunks_for_file(self, file) -> int:
        if self._batch_by == "files" or self._chunk_length is None:
            return 1
        return max(1, round(file['duration'] * self.sr / self.chunk_length))

    def _build_chunk_map(self) -> None:
        self._cum_chunks = []
        n_chunks = 0
        for file_idx in self.file_indices:
            n_chunks += self._n_chunks_for_file(self._files[file_idx])
            self._cum_chunks.append(n_chunks)

    def configure(self, chunk_length: int | float | None = None, batch_by: BatchByType | None = None, pad_mode: PadMode | str | None = None, 
                  features_after_transforms: bool | None = None, **kwargs) -> "OnlineAudioDataset":
        """returns a copy of the dataset with given loading parameters."""
        dataset = copy.copy(super().configure(features_after_transforms=features_after_transforms, **kwargs))
        dataset._set_loading_params(chunk_length=chunk_length, batch_by=batch_by, pad_mode=pad_mode)
        return dataset

    def _check_loading_params(self):
        if self._cum_chunks is None:
            self._build_chunk_map()

    def _check_loader_arguments(self, batch_size, loader_kwargs):
        if self._chunk_length is None and batch_size is not None and batch_size > 1 and loader_kwargs.get('collate_fn') is None:
            raise ValueError("items of OnlineAudioDataset are whole files if chunk_length is None, and cannot be batched : "
                             "give chunk_length (in constructor, configure, or get_loader), batch_size=1, or a collate_fn")

    def __len__(self):
        self._check_loading_params()
        return self._cum_chunks[-1] if len(self._cum_chunks) > 0 else 0

    def chunk_position(self, index: int, random_offset: bool = True):
        """returns (file index, start position in seconds) of a given item. If random_offset is False, the 
        chunk is taken at the beginning of its slot."""
        self._check_loading_params()
        if index < 0: index += len(self)
        if not (0 <= index < len(self)):
            raise IndexError("index %d out of range for dataset of length %d"%(index, len(self)))
        position = bisect.bisect_right(self._cum_chunks, index)
        file_idx = self.file_indices[position]
        if self._chunk_length is None:
            return file_idx, 0.
        n_chunks = self._n_chunks_for_file(self._files[file_idx])
        chunk_idx = index - (self._cum_chunks[position - 1] if position > 0 else 0)
        duration = self._files[file_idx]['duration']
        chunk_duration = self.chunk_length / self.sr
        # each chunk is sampled in its slot, without exceeding the file if possible
        slot_length = duration / n_chunks
        max_start = max(0., duration - chunk_duration)
        slot_start = min(chunk_idx * slot_length, max_start)
        slot_end = min(slot_start + max(0., slot_length - chunk_duration), max_start)
        start = random.uniform(slot_start, slot_end) if random_offset else slot_start
        return file_idx, start

    def _fragment_from_index(self, index):
        return self._fragment_from_audio(*self._load_chunk(index))

    def get(self, index, output_pattern=None, transforms=None, _global_index=None):
        """returns the item at index (see AudioDataset.get). If features_after_transforms (default), features are computed
        on the augmented audio, i.e. on the waveform output after its transforms (see batch.augmented_audio_entry) ; other
        waveform entries (e.g. clean targets) are given without the transforms of this entry. Otherwise, features are
        computed on loaded audio, before transforms."""
        transforms = transforms or self.transforms
        if is_multi_index(index) or transforms is None or len(self._features) == 0 or not self.features_after_transforms:
            return super().get(index, output_pattern=output_pattern, transforms=transforms, _global_index=_global_index)
        kind, entries = parse_output_pattern(output_pattern or self.output_pattern)
        entry_transforms = transforms_per_entry(kind, entries, transforms)
        audio_entry = augmented_audio_entry(entries, entry_transforms)
        if audio_entry is None or entry_transforms[audio_entry] is None:
            # no audio augmentation : features are computed on loaded audio
            return super().get(index, output_pattern=output_pattern, transforms=transforms, _global_index=_global_index)
        if hasattr(index, "item") and not isinstance(index, (str, bytes)):
            index = index.item()
        item_index = _global_index if _global_index is not None else (index % len(self) if isinstance(index, int) else None)
        file_idx, start, wav = self._load_chunk(index)
        clean = self._fragment_from_audio(file_idx, start, wav, features=[])
        audio = _entry_from_fragment(clean, entries[audio_entry][0], item_index)
        audio = entry_transforms[audio_entry](_check_channels(audio, self._n_channels))
        augmented_wav = torch.as_tensor(np.ascontiguousarray(audio) if isinstance(audio, np.ndarray) else audio).float()
        augmented = self._fragment_from_audio(file_idx, start, augmented_wav.reshape(-1, augmented_wav.shape[-1]))
        outs = []
        for i, (source, _) in enumerate(entries):
            if i == audio_entry:
                outs.append(audio)
                continue
            # other waveform entries are taken from loaded audio, features and metadata from augmented audio
            out = _entry_from_fragment(clean if entry_name(source) == "waveform" else augmented, source, item_index)
            t = entry_transforms[i]
            outs.append(out if t is None else t(_check_channels(out, self._n_channels)))
        if kind == "single":
            return outs[0]
        if kind == "tuple":
            return tuple(outs)
        return {alias: out for out, (_, alias) in zip(outs, entries)}

    def _load_chunk(self, index):
        """returns (file index, start, audio) of the chunk at index (silent chunks are drawn again, see loudness_threshold)."""
        file_idx, start = self.chunk_position(index)
        wav = self.load_audio(file_idx, start=start, length=self.chunk_length)
        threshold = getattr(self, "_loudness_threshold", None)
        if threshold is not None and self._chunk_length is not None:
            file = self._files[file_idx]
            max_start = max(0., file['duration'] - self.chunk_length / self.sr)
            for _ in range(self._max_retries):
                if self._chunk_loudness(wav) >= threshold:
                    break
                # silent chunks are drawn again at a random position of the file
                start = random.uniform(0., max_start)
                wav = self.load_audio(file_idx, start=start, length=self.chunk_length)
        return file_idx, start, wav

    def _chunk_loudness(self, wav: torch.Tensor) -> float:
        from ..utils import loudness
        return float(loudness(wav[..., :5, :] if wav.shape[-2] > 5 else wav, self.sr))

    def item_labels(self, features, n_bins: int = 10):
        """labels of items : values of path-based features of their files, and histogram bins of computed features 
        (averaged over the items of each file, see _file_feature_mean)."""
        from ..sampling import bin_values
        self._check_loading_params()
        computed = {f.feature_name: f for f in self._features}
        files = self.file_indices
        n_items = [self._cum_chunks[p] - (self._cum_chunks[p - 1] if p > 0 else 0) for p in range(len(files))]
        per_feature = []
        for f in checklist(features):
            if f in self.path_features:
                file_labels = [self._file_feature_value(i, f) for i in files]
            elif f in computed:
                file_labels = bin_values([self._file_feature_mean(i, computed[f]) for i in files], n_bins=n_bins)
            else:
                raise ValueError("feature %s cannot be used for labels ; available features : %s"%(f, self.path_features + list(computed.keys())))
            per_feature.append([l for l, n in zip(file_labels, n_items) for _ in range(n)])
        return self._combine_labels(per_feature)

    def _computed_features(self):
        return {f.feature_name: f for f in self._features}

    def _without_features(self, names):
        dataset = copy.copy(self)
        dataset._features = [f for f in self._features if f.feature_name not in names]
        return dataset

    @property
    def loudness_threshold(self) -> float | None:
        return getattr(self, "_loudness_threshold", None)

    # loading

    def _adapt_channels(self, wav: torch.Tensor) -> torch.Tensor:
        channels = self._n_channels
        if channels is None:
            return wav
        if wav.shape[0] > channels:
            wav = wav[:channels]
        elif wav.shape[0] < channels:
            wav = wav[torch.arange(channels) % wav.shape[0]]
        return wav

    def load_audio(self, file_idx: int, start: float = 0., length: int | float | None = None, pad_mode: PadMode | str | None = None) -> torch.Tensor:
        """loads a chunk of a given file, resampled to target sr and adapted to target channels.

        Args:
            file_idx (int): file index (in self.files).
            start (float, optional): start position (seconds). Defaults to 0.
            length (int | float | None, optional): chunk length, in samples (int, at target sr) or seconds (float). 
                None for the whole file. Defaults to None.
            pad_mode (PadMode | str | None, optional): padding mode if the chunk exceeds the file ; None for dataset's pad_mode. Defaults to None.
        """
        file = self._files[file_idx]
        orig_sr = file['sr']
        target_sr = self._sr or orig_sr
        frame_offset = int(round(start * orig_sr))
        target_length = None if length is None else chunk_length_to_samples(length, target_sr)
        num_frames = -1 if length is None else int(math.ceil(target_length * orig_sr / target_sr))
        # resampled chunks are loaded with surrounding context, such that they match chunks of resampled files (as in 
        # pre-processing). Left context is a multiple of the resampling period, to be trimmed exactly after resampling.
        left_context, right_context = 0, 0
        if orig_sr != target_sr:
            period = orig_sr // math.gcd(orig_sr, target_sr)
            context = int(math.ceil(self._resampling_context / period)) * period
            left_context = min(context, (frame_offset // period) * period)
            right_context = context if num_frames >= 0 else 0
        wav, orig_sr = load_audio_range(file['path'], frame_offset=frame_offset - left_context, 
                                        num_frames=num_frames + left_context + right_context if num_frames >= 0 else -1, 
                                        backend=file.get('backend'))
        if orig_sr != target_sr and wav.shape[-1] > 0:
            wav = torchaudio.functional.resample(wav, orig_sr, target_sr)
            wav = wav[..., left_context * target_sr // orig_sr:]
        wav = self._adapt_channels(wav)
        if target_length is not None:
            if wav.shape[-1] > target_length:
                wav = wav[..., :target_length]
            elif wav.shape[-1] < target_length:
                pad_mode = pad_mode or self._pad_mode
                if isinstance(pad_mode, str): pad_mode = getattr(PadMode, pad_mode.upper())
                wav = pad(wav, target_length, pad_mode)
        return wav

    def get_fragment(self, file_idx: int, start: float = 0., length: float | None = None, features: Iterable[AcidsDatasetFeature] | None = None, **kwargs):
        """loads a chunk of a given file as a fragment, with path-based metadata and dynamically computed features.

        Args:
            file_idx (int): file index (in self.files).
            start (float, optional): start position (seconds). Defaults to 0.
            length (int | float | None, optional): chunk length, in samples (int, at target sr) or seconds (float). 
                None for the whole file. Defaults to None.
            features (Iterable[AcidsDatasetFeature] | None, optional): features to compute ; None for dataset features. Defaults to None.
        """
        wav = self.load_audio(file_idx, start=start, length=length, **kwargs)
        return self._fragment_from_audio(file_idx, start, wav, features=features)

    def _fragment_from_audio(self, file_idx: int, start: float, wav: torch.Tensor, features=None):
        file = self._files[file_idx]
        return self._build_fragment(self._fragment_class, wav, self._sr or file['sr'], file['audio_path'], start, file['duration'], 
                                    self._features if features is None else features, output_type=self._output_type, 
                                    bformat=self._bformat, metadata=file['metadata'])

    # baked features

    def _baked_roots(self, path: str | Path | None = None) -> Dict[str, Path]:
        """{dataset folder name: baked folder} : an "ad-features" sub-folder of each dataset folder, or a folder named
        after each dataset folder in path."""
        if path is None:
            return {p.name: p / BAKED_FOLDER for p in self._paths}
        path = Path(path).resolve()
        return {p.name: path / p.name for p in self._paths}

    @property
    def baked_features(self) -> List[str]:
        """names of features read from baked files (see use_baked_features)."""
        return [f.feature_name for f in self._features if isinstance(f, BakedFeature)]

    def _baked_file_info(self, file_idx: int) -> Dict[str, Any]:
        file = self._files[file_idx]
        return {'audio_stamp': {'size': file.get('size'), 'mtime': file.get('mtime')}, 'original_sr': file['sr'],
                'sr': self._sr or file['sr'], 'channels': self._n_channels}

    def _is_baked_file_valid(self, metadata: Dict[str, Any], file_idx: int) -> bool:
        """whether a baked file matches the audio file (unchanged) and the loading parameters (sr, channels)."""
        expected = self._baked_file_info(file_idx)
        return metadata.get('version') == BAKED_VERSION and all(metadata.get(k) == expected[k] for k in ['audio_stamp', 'sr', 'channels'])

    def bake_features(self, *features: str | AcidsDatasetFeature, path: str | Path | None = None, overwrite: bool = False) -> "OnlineAudioDataset":
        """computes features on whole audio files, and stores them in .ad files, such that they are read instead of
        computed when loading chunks (see use_baked_features). Baked files mirror the architecture of dataset folders :
        <dataset folder>/ad-features/<relative audio path>.ad, or <path>/<dataset folder name>/<relative audio path>.ad
        if path is given. They contain the features of the whole file (time last) and their metadata : original
        sampling rate of the file, sampling rate and channels of the audio features are computed on, frame rate
        (feature_sr) and frame offset of each feature, and the configuration of features.

        Features must have a time axis (last) : per-chunk descriptors (e.g. Tempo) and non-array features (e.g.
        AfterMIDI) cannot be baked. Baking is incremental : features already baked with the same configuration, for
        unchanged audio files, are skipped (unless overwrite), and new features are added to existing files.

        Args:
            *features (str | AcidsDatasetFeature): names of dataset features, or feature objects ; every feature of
                the dataset if none is given.
            path (str | Path | None, optional): destination folder ; None for "ad-features" sub-folders of dataset
                folders. Defaults to None.
            overwrite (bool, optional): re-compute features already baked. Defaults to False.

        Returns:
            OnlineAudioDataset: the dataset (features are still computed dynamically ; see use_baked_features).
        """
        available = {f.feature_name: f for f in self._features if not isinstance(f, BakedFeature)}
        if len(features) == 0:
            features = list(available.values())
        to_bake = []
        for f in features:
            if isinstance(f, AcidsDatasetFeature):
                to_bake.append(f)
            elif f in available:
                to_bake.append(available[f])
            else:
                raise ValueError("feature %s cannot be baked : available features are %s"%(f, list(available)))
        if len(to_bake) == 0:
            raise ValueError("no feature to bake")
        roots = self._baked_roots(path)
        frame_rates = {}
        for file_idx in self.file_indices:
            file = self._files[file_idx]
            baked_path = baked_file_path(roots, file['audio_path'])
            metadata, arrays = {'features': {}}, {}
            if baked_path.exists():
                try:
                    metadata, arrays = read_baked_file(baked_path)
                    if not self._is_baked_file_valid(metadata, file_idx):
                        metadata, arrays = {'features': {}}, {}
                except Exception as e:
                    logging.warning("could not read baked file %s (%s) ; overwriting"%(baked_path, e))
                    metadata, arrays = {'features': {}}, {}
            missing = [f for f in to_bake if overwrite or metadata['features'].get(f.feature_name, {}).get('feature') != str(f)]
            if len(missing) == 0:
                continue
            wav = self.load_audio(file_idx)
            sr = self._sr or file['sr']
            fragment = self._fragment_from_audio(file_idx, 0., wav, features=missing)
            for f in missing:
                if (f.feature_name, sr) not in frame_rates:
                    frame_rates[(f.feature_name, sr)] = self._probe_baked_feature(f, sr, wav.shape[0])
                feature_sr, frame_offset = frame_rates[(f.feature_name, sr)]
                arrays[f.feature_name] = to_numpy(fragment.get(f.feature_name))
                metadata['features'][f.feature_name] = {'feature_sr': feature_sr, 'frame_offset': frame_offset, 'feature': str(f),
                                                        'shape': list(arrays[f.feature_name].shape)}
            metadata.update(version=BAKED_VERSION, audio_path=file['audio_path'], n_samples=int(wav.shape[-1]),
                            **self._baked_file_info(file_idx))
            metadata['channels'] = self._n_channels
            write_baked_file(baked_path, metadata, arrays)
        logging.info("baked features %s in %s"%([f.feature_name for f in to_bake], sorted(set(str(r) for r in roots.values()))))
        return self

    def _probe_baked_feature(self, feature: AcidsDatasetFeature, sr: int, n_channels: int):
        """frame rate and frame offset of a feature, computed on test signals."""
        generator = torch.Generator().manual_seed(0)
        def compute(n_samples):
            wav = 0.1 * torch.randn(n_channels, n_samples, generator=generator)
            fragment = self._build_fragment(self._fragment_class, wav, sr, "probe/probe.wav", 0., n_samples / sr, [feature],
                                            output_type="numpy", bformat=self._bformat)
            return np.atleast_1d(to_numpy(fragment.get(feature.feature_name)))
        try:
            return probe_frame_rate(compute, sr, n_channels, hop_hint=hop_hint(feature))
        except Exception as e:
            raise ValueError("feature %s cannot be baked : %s"%(feature.feature_name, e)) from e

    def use_baked_features(self, *feature_names: str, path: str | Path | None = None, check: bool = True) -> "OnlineAudioDataset":
        """reads given features from baked files (see bake_features) instead of computing them : chunks get the frames
        of their position in the features of whole files. Features of the dataset with the same names are replaced ;
        other names are added to dataset features. For a chunk starting at t seconds, frames start at the frame closest
        to t : values may differ from features computed on chunks by the alignment of frames (up to half a hop) and
        by edge effects (features of chunks are computed on padded chunks, baked ones on whole files).

        Args:
            *feature_names (str): names of baked features ; every feature baked for the first file if none is given.
            path (str | Path | None, optional): folder of baked files given to bake_features ; None for "ad-features"
                sub-folders of dataset folders. Defaults to None.
            check (bool, optional): checks that every file of the dataset has an up-to-date baked file (unchanged audio
                file, same sampling rate and channels) containing the features, with the same configuration as dataset
                features of the same name ; otherwise, missing baked files raise errors when loading. Defaults to True.

        Returns:
            OnlineAudioDataset: the dataset.
        """
        roots = self._baked_roots(path)
        if len(feature_names) == 0:
            first_path = baked_file_path(roots, self._files[self.file_indices[0]]['audio_path'])
            if not first_path.exists():
                raise FileNotFoundError("no baked file found at %s : bake features with bake_features"%first_path)
            feature_names = list(read_baked_metadata(first_path)['features'])
        if check:
            current = {f.feature_name: str(f) for f in self._features if not isinstance(f, BakedFeature)}
            problems = []
            for file_idx in self.file_indices:
                file = self._files[file_idx]
                baked_path = baked_file_path(roots, file['audio_path'])
                if not baked_path.exists():
                    problems.append("%s : missing"%baked_path); continue
                metadata = read_baked_metadata(baked_path)
                if not self._is_baked_file_valid(metadata, file_idx):
                    problems.append("%s : outdated (audio file modified, or different sr / channels)"%baked_path); continue
                for name in feature_names:
                    if name not in metadata['features']:
                        problems.append("%s : feature %s not baked"%(baked_path, name))
                    elif name in current and metadata['features'][name]['feature'] != current[name]:
                        problems.append("%s : feature %s baked as %s, but dataset feature is %s"%(baked_path, name, metadata['features'][name]['feature'], current[name]))
            if len(problems) > 0:
                raise ValueError("%d problem(s) found with baked features (use bake_features to bake them again, or check=False) :\n%s"%(
                    len(problems), "\n".join(problems[:10]) + ("\n..." if len(problems) > 10 else "")))
        baked = {name: BakedFeature(name, roots) for name in feature_names}
        features = [baked.pop(f.feature_name) if f.feature_name in baked else f for f in self._features]
        self._features = features + list(baked.values())
        return self

    # feature search
    # path-based features (original_path and meta_regexp) are known for every file without loading audio ;
    # searching and partitioning are thus performed on files, such that chunks of a given file always 
    # belong to the same partition.

    @property
    def path_features(self) -> List[str]:
        """names of features available for search and partitioning."""
        return ['original_path'] + [f.feature_name for f in self._regexp_features]

    def _file_feature_value(self, file_idx: int, feature_name: str):
        file = self._files[file_idx]
        if feature_name == "original_path":
            return file['audio_path']
        value = file['metadata'].get(feature_name)
        return tuple(value) if isinstance(value, list) else value

    def _check_path_features(self, features: Iterable[str]) -> None:
        for f in features:
            if f not in self.path_features:
                raise ValueError("feature %s cannot be used for search or partitioning ; available features : %s"%(f, self.path_features))

    @property
    def feature_hash(self) -> Dict[str, Dict[Any, List[int]]]:
        """{feature_name: {value: [file indices]}} for path-based features of dataset files."""
        feature_hash = {f: {} for f in self.path_features}
        for i in self.file_indices:
            for f in feature_hash:
                value = self._file_feature_value(i, f)
                if value is not None:
                    feature_hash[f].setdefault(value, []).append(i)
        return feature_hash

    def index_for_features(self, **kwargs) -> Dict[tuple, set]:
        """returns {(value_1, ..., value_n): set of file indices} for given values of path-based features 
        (e.g. index_for_features(instrument=["piano", "guitar"], idx=0))."""
        self._check_path_features(kwargs.keys())
        feature_hash = self.feature_hash
        idx_dict = {(): set(self.file_indices)}
        for feature_name, feature_value in kwargs.items():
            new_idx_dict = {}
            for v in checklist(feature_value):
                # regexp values are strings
                hash_tmp = feature_hash[feature_name].get(v, feature_hash[feature_name].get(str(v)))
                if not hash_tmp: 
                    raise IndexError("could not find value %s for feature %s"%(v, feature_name))
                for kk, vv in idx_dict.items():
                    new_idx_dict[kk + (v,)] = vv.intersection(hash_tmp)
            idx_dict = new_idx_dict
        return idx_dict

    def item_indices_for_files(self, file_indices: Iterable[int]) -> List[int]:
        """returns dataset item indices corresponding to given files."""
        self._check_loading_params()
        positions = {f: p for p, f in enumerate(self.file_indices)}
        items = []
        for f in file_indices:
            if f not in positions: 
                raise IndexError("file %d not in dataset"%f)
            p = positions[f]
            items.extend(range(self._cum_chunks[p - 1] if p > 0 else 0, self._cum_chunks[p]))
        return items

    def query_from_features(self, n: int | EllipsisType, _random: bool = True, **kwargs) -> Dict[tuple, List[Any]]:
        """returns {(value_1, ..., value_n): [items]} for given values of path-based features. n is the number of 
        items per entry (Ellipsis for all items) ; if _random, items are randomly chosen (with replacement)."""
        assert isinstance(n, (int, EllipsisType)), "n must be either an int or Ellipsis"
        idx_dict = self.index_for_features(**kwargs)
        data_dict = {}
        for k, file_indices in idx_dict.items():
            items = self.item_indices_for_files(sorted(file_indices))
            if n is not Ellipsis and len(items) > 0:
                items = random.choices(items, k=n) if _random else items[:n]
            data_dict[k] = [self[i] for i in items]
        return data_dict

    # partitioning

    _default_partition_name = "partitions"

    @property
    def partition_dir(self) -> Path:
        if self._partition_dir is not None:
            return self._partition_dir
        if self._index_dir is not None:
            return self._index_dir / "partitions"
        return self._paths[0] / ".ad_partitions"

    def _partition_path(self, name: str | bool | None) -> Path:
        if name is None or name is True:
            name = self._default_partition_name
        if os.path.splitext(name)[1] != ".json": 
            name += ".json"
        return self.partition_dir / name

    def get_subdataset(self, subindices: Iterable[int], check: bool = False) -> "OnlineAudioDataset":
        """returns a dataset restricted to given file indices."""
        subindices = sorted(set(int(i) for i in subindices))
        if check:
            for i in subindices:
                if not (0 <= i < len(self._files)): 
                    raise IndexError("file index %d out of range"%i)
        subdataset = copy.copy(self)
        subdataset._subindices = subindices
        subdataset.parent = self
        subdataset._build_chunk_map()
        return subdataset

    def _file_weights(self, file_indices: List[int]) -> List[float]:
        # number of items if chunks are defined, durations otherwise
        if self._chunk_length is not None:
            return [float(self._n_chunks_for_file(self._files[i])) for i in file_indices]
        return [self._files[i]['duration'] for i in file_indices]

    def _split_without_feature(self, partitions: Dict[str, float], rng=random) -> Dict[str, List[int]]:
        files = rng.sample(self.file_indices, k=len(self.file_indices))
        return self._assign_by_ratio(files, self._file_weights(files), partitions)

    def _sort_split_features(self, features: List[str]) -> tuple:
        """separates path-based (categorical) features from computed (continuous) features."""
        computed = [f.feature_name for f in self._features]
        categorical, continuous = [], []
        for f in features:
            if f in self.path_features: categorical.append(f)
            elif f in computed: continuous.append(f)
            else: raise ValueError("feature %s cannot be used for partitioning ; available features : %s"%(f, self.path_features + computed))
        if len(continuous) > 1:
            raise ValueError("partitions can only be stratified over one continuous feature, got %s"%continuous)
        return categorical, (continuous[0] if continuous else None)

    def _file_feature_mean(self, file_idx: int, feature) -> float:
        """mean of a computed feature over the items of a file (taken at the beginning of their slots, or whole files 
        if chunk_length is None)."""
        if self._chunk_length is None:
            starts = [0.]
        else:
            file = self._files[file_idx]
            n_chunks = self._n_chunks_for_file(file)
            starts = [min(i * file['duration'] / n_chunks, max(0., file['duration'] - self.chunk_length / self.sr)) for i in range(n_chunks)]
        values = [self._reduce_feature_value(self.get_fragment(file_idx, start=s, length=self.chunk_length, features=[feature]).get(feature.feature_name)) 
                  for s in starts]
        values = [v for v in values if np.isfinite(v)]
        return float(np.mean(values)) if values else float("nan")

    def _split_stratified(self, partitions, groups: Dict[Any, List[int]], feature_name: str, n_bins: int = 10, rng=random) -> Dict[str, List[int]]:
        feature = next(f for f in self._features if f.feature_name == feature_name)
        labels, weights, values = [], [], []
        n_missing = 0
        for label, files in groups.items():
            file_weights = self._file_weights(files)
            file_values = [self._file_feature_mean(i, feature) for i in files]
            valid = [(w, v) for w, v in zip(file_weights, file_values) if np.isfinite(v)]
            if not valid:
                n_missing += len(files)
                continue
            labels.append(label)
            weights.append(sum(file_weights))
            values.append(sum(w * v for w, v in valid) / sum(w for w, _ in valid))
        if n_missing > 0:
            logging.warning("%d files have no finite value for feature %s, and are not partitioned"%(n_missing, feature_name))
        assignment = self._assign_stratified(labels, weights, values, partitions, n_bins=n_bins, rng=rng)
        return {k: sorted(sum([groups[l] for l in v], [])) for k, v in assignment.items()}

    def _split_with_features(self, partitions: Dict[str, float], features: List[str], tolerance: float = 0.1, balance_cardinality: bool = False, rng=random, n_bins: int = 10) -> Dict[str, List[int]]:
        features, continuous = self._sort_split_features(features)
        if continuous is not None and len(features) == 0:
            # files are stratified over the continuous feature
            return self._split_stratified(partitions, {(i,): [i] for i in self.file_indices}, continuous, n_bins=n_bins, rng=rng)
        self._check_path_features(features)
        groups = {}
        n_missing = 0
        for i in self.file_indices:
            label = tuple(self._file_feature_value(i, f) for f in features)
            if None in label:
                n_missing += 1
                continue
            groups.setdefault(label, []).append(i)
        if n_missing > 0:
            logging.warning("%d files have no value for features %s, and are not partitioned"%(n_missing, features))
        if continuous is not None:
            # groups of categorical features are stratified over the continuous feature
            return self._split_stratified(partitions, groups, continuous, n_bins=n_bins, rng=rng)
        labels = rng.sample(list(groups.keys()), k=len(groups))
        if balance_cardinality:
            weights = [sum(self._file_weights(groups[l])) for l in labels]
            assignment = self._assign_balanced(labels, weights, partitions, tolerance=tolerance)
        else:
            assignment = self._assign_by_ratio(labels, [1.] * len(labels), partitions)
        return {k: sum([groups[l] for l in v], []) for k, v in assignment.items()}

    def _split_from_feature(self, partition_key: str, partitions: Iterable[str]) -> Dict[str, List[int]]:
        feature_hash = self.feature_hash
        assert partition_key in feature_hash, "key %s not found in feature_hash"%(partition_key)
        for p in partitions:
            assert p in feature_hash[partition_key], "partition %s not found in %s"%(p, partition_key)
        return {p: feature_hash[partition_key][p] for p in partitions}

    def _write_partition(self, partitions: Dict[str, "OnlineAudioDataset"], write: str | bool) -> None:
        partition_path = self._partition_path(write)
        os.makedirs(partition_path.parent, exist_ok=True)
        # files are referred with their audio_path, such that partitions are robust to dataset modifications
        with open(partition_path, "w+") as f:
            json.dump({p: [self._files[i]['audio_path'] for i in v.file_indices] for p, v in partitions.items()}, f, indent=1)

    def has_partition(self, name: str) -> bool:
        return self._partition_path(name).exists()

    def split(self, 
              partitions: Dict[str, float], 
              features: List[str] | None = None, 
              use_meta_if_available: str | bool | None = True,
              load_if_available: bool | str | None = False,
              balance_cardinality: bool = False, 
              tolerance: float = 0.1, 
              write: str | bool | None = None,
              seed: int | None = None,
              n_bins: int = 10) -> Dict[str, "OnlineAudioDataset"]:
        """splits dataset files in partitions.

        Args:
            partitions (Dict[str, float]): partition ratios (e.g. {"train": 0.8, "valid": 0.2}).
            features (List[str] | None, optional): path-based features (original_path, or meta_regexp features) used to group 
                files, such that each feature value belongs to a single partition. A computed (continuous) feature can also be 
                given (e.g. loudness) : partitions are then stratified over its values (see n_bins). Defaults to None.
            use_meta_if_available (str | bool | None, optional): if a path-based feature with this name (or "partition" if True) 
                exists, its values are used as partitions. Defaults to True.
            load_if_available (bool | str | None, optional): loads the partition with this name (or default name if True) 
                if it exists. Defaults to False.
            balance_cardinality (bool, optional): balance partitions sizes rather than number of feature values. Defaults to False.
            tolerance (float, optional): tolerance for balanced partitions. Defaults to 0.1.
            write (str | bool | None, optional): writes partition with this name (or default name if True). Defaults to None.
            seed (int | None, optional): seed of random partitions ; partitions with the same seed are identical (e.g. across 
                processes of distributed training). If None, the global random state is used. Defaults to None.
            n_bins (int, optional): number of histogram (quantile) bins of continuous features : files are shuffled within 
                bins, and bins are distributed among partitions, such that each partition has a similar distribution of the 
                feature. Feature values are computed on the items of each file. Defaults to 10.
        """
        assert not self.is_partition, "dataset is already a partition of an existing dataset."
        rng = random.Random(seed) if seed is not None else random
        if load_if_available:
            name = None if load_if_available is True else load_if_available
            if self.has_partition(name):
                if features is not None: logging.warning("found partition to load, but features was not None; feature-based partitioning ignored")
                return self.load_partition(name)

        file_partitions = None
        if use_meta_if_available:
            partition_key = "partition" if use_meta_if_available is True else use_meta_if_available
            if partition_key in self.path_features:
                try: 
                    file_partitions = self._split_from_feature(partition_key, partitions.keys())
                    if features is not None: logging.warning("found partition feature, but features was not None; feature-based partitioning ignored")
                except AssertionError as e:
                    logging.warning('Could not find partition using feature %s (%s). Creating new partition'%(partition_key, e))
        if file_partitions is None:
            if features is None:
                file_partitions = self._split_without_feature(partitions, rng=rng)
            else:
                file_partitions = self._split_with_features(partitions, checklist(features), tolerance=tolerance, balance_cardinality=balance_cardinality, rng=rng, n_bins=n_bins)

        partitions = {k: self.get_subdataset(v) for k, v in file_partitions.items()}
        if write: 
            self._write_partition(partitions, write)
        return partitions

    def load_partition(self, name: str | None = None, check: bool = False) -> Dict[str, "OnlineAudioDataset"]:
        partition_path = self._partition_path(name)
        if not partition_path.exists():
            raise FileNotFoundError("could not load partition %s"%partition_path)
        with open(partition_path, "r") as f:
            partition_paths = json.load(f)
        file_map = {f['audio_path']: i for i, f in enumerate(self._files)}
        partitions = {}
        for p, audio_paths in partition_paths.items():
            missing = [a for a in audio_paths if a not in file_map]
            if len(missing) > 0:
                if check: raise FileNotFoundError("partition %s : files %s not found in dataset"%(p, missing))
                logging.warning("partition %s : %d files not found in dataset"%(p, len(missing)))
            partitions[p] = self.get_subdataset([file_map[a] for a in audio_paths if a in file_map])
        return partitions


    # conversion to pre-processed datasets

    def list_partitions(self) -> List[str]:
        """names of written partitions."""
        return sorted(f.stem for f in Path(self.partition_dir).glob("*.json")) if Path(self.partition_dir).is_dir() else []

    def _estimate_db_size(self, files, hop_samples: int, waveform: bool) -> float:
        """size (Gb) allocated to the pre-processed database : int16 audio of overlapping chunks, with a margin for features."""
        duration = sum(f['duration'] for f in files)
        channels = self._n_channels or max(f['channels'] for f in files)
        audio_bytes = duration * self.sr * channels * 2 * (self.chunk_length / hop_samples) if waveform else 0.
        return max(1., 4 * audio_bytes / 1024 ** 3)

    def preprocess(self, out: str | Path, hop_length: int | float | None = None, partitions: bool | List[str] | None = True, 
                   exclude_flags: List[str] | None = None, health_thresholds: Dict[str, float] | None = None, features = None, 
                   waveform: bool = True, preset: str = "default", override: List[str] | None = None, max_db_size: float | None = None, 
                   force: bool = False, compact: bool = False, device: str | None = None, check: bool = False, **dataset_kwargs: Any):
        """compiles the dataset into a pre-processed dataset, with the same files, parsing settings and features, and returns 
        it (as a PreprocessedAudioDataset, with the output pattern and transforms of this dataset).

        Args:
            out (str | Path): folder of the pre-processed dataset.
            hop_length (int | float | None, optional): hop between pre-processed chunks, in samples (int) or seconds (float) ; 
                half of chunk_length if None (as preprocess_dataset). Defaults to None.
            partitions (bool | List[str] | None, optional): written partitions carried to the pre-processed dataset (all of 
                them if True), matching files by audio paths. Defaults to True.
            exclude_flags (List[str] | None, optional): files with these health flags (e.g. ["silent", "clipping"], see 
                acids_dataset.health) are excluded ; health is computed on every file. Defaults to None.
            features (optional): features computed in the pre-processed dataset ; the features of this dataset if None.
            waveform (bool, optional): store audio in the pre-processed dataset. Defaults to True.
            preset (str, optional): preset of other parsing parameters (sampling rate, channels and chunk length are the ones of 
                this dataset). Defaults to "default".
            override (List[str] | None, optional): additional gin bindings. Defaults to None.
            max_db_size (float | None, optional): size of the database (Gb) ; estimated from durations if None. Defaults to None.
            **dataset_kwargs: additional arguments of the returned dataset.

        Files known as unreadable by the index are not parsed. Chunks are padded with the padding mode of this dataset, and 
        chunks below its loudness_threshold are discarded. The source of the dataset is recorded in its metadata.
        """
        from ..preprocess import preprocess_dataset
        from .simple import PreprocessedAudioDataset
        if self._chunk_length is None:
            raise ValueError("pre-processed datasets require a chunk_length (give it in constructor or configure)")
        if self.sr is None:
            raise ValueError("files have different sampling rates ; please provide sr to dataset")
        files = [self._files[i] for i in self.file_indices]
        excluded = []
        if exclude_flags:
            from ..health import dataset_health
            report = dataset_health(self, thresholds=health_thresholds)
            flagged = {r['audio_path'] for r in report['files'] if set(r['flags']) & set(exclude_flags)}
            excluded = sorted(f['audio_path'] for f in files if f['audio_path'] in flagged)
            files = [f for f in files if f['audio_path'] not in flagged]
        if len(files) == 0:
            raise ValueError("no file to pre-process")
        hop_samples = chunk_length_to_samples(hop_length, self.sr) if hop_length is not None else self.chunk_length // 2
        overrides = ["RawParser.pad_mode = '%s'"%self._pad_mode.name.lower()]
        if self.loudness_threshold is not None:
            overrides.append("RawParser.loudness_threshold = %r"%self.loudness_threshold)
        source = {'type': "online", 'paths': [str(p) for p in self._paths], 
                  'indexes': [str(self.index_path(p)) for p in self._paths if self.index_path(p).exists()], 
                  'filters': list(self._filters), 'exclude': list(self._exclude), 'meta_regexp': list(self._meta_regexp), 
                  'excluded_files': excluded, 'invalid_files': list(self.invalid_files), 'pad_mode': self._pad_mode.name.lower(), 
                  'loudness_threshold': self.loudness_threshold}
        preprocess_dataset([str(p) for p in self._paths], out=out, preset=preset, chunk_length=int(self.chunk_length), hop_length=int(hop_samples), 
                           sample_rate=int(self.sr), channels=self._n_channels, features=list(self._features if features is None else features), 
                           meta_regexp=list(self._meta_regexp), waveform=waveform, override=overrides + list(override or []), 
                           max_db_size=max_db_size or self._estimate_db_size(files, hop_samples, waveform), force=force, compact=compact, 
                           check=check, device=device, files=[f['path'] for f in files], metadata={'source': source})
        # arguments of the returned dataset default to the ones of this dataset
        dataset_kwargs = {'output_pattern': self.output_pattern, 'transforms': self.transforms, 
                          'channels': self._n_channels if self._n_channels is not None else 1, **dataset_kwargs}
        dataset = PreprocessedAudioDataset(out, **dataset_kwargs)
        names = self.list_partitions() if partitions is True else list(partitions or [])
        for name in names:
            self._transfer_partition(name, dataset)
        return dataset

    def _transfer_partition(self, name: str, dataset) -> None:
        """writes a partition of this dataset in a pre-processed dataset, matching files by audio paths."""
        with open(self._partition_path(name), "r") as f:
            file_partitions = json.load(f)
        path_keys = dataset.feature_hash['original_path']
        partitions = {}
        for p, audio_paths in file_partitions.items():
            keys = sorted({dataset._key_to_str(k) for a in audio_paths for k in path_keys.get(a, [])})
            partitions[p] = dataset.get_subdataset([dataset._loader.keygen.from_str(k) for k in keys])
        dataset._write_partition(partitions, name)


__all__ = ['OnlineAudioDataset']
