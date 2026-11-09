"""Backend of the dataset browser : wraps an AudioDataset (pre-processed or online) and gives everything the user
interface needs (summary, files, items, feature statistics, augmentations, partitions), independently of the UI
library."""
import os
import math
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..datasets import AudioDataset, OnlineAudioDataset, PreprocessedAudioDataset, is_preprocessed_path


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        # JSON cannot encode infinite values (e.g. levels of silent files)
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class DatasetBrowser():
    """gives access to a dataset for exploration (see acids_dataset.ui.launch)."""

    def __init__(self, dataset: AudioDataset):
        self.dataset = dataset
        self._transform_cache = {}

    @classmethod
    def from_path(cls, path, chunk_length: int | float | None = None, write_index: bool | None = False, **kwargs) -> "DatasetBrowser":
        """loads a pre-processed dataset or a raw audio folder. Online datasets are given a default chunk length of
        one second, and the most common sampling rate of their files if they have several."""
        if is_preprocessed_path(path):
            return cls(AudioDataset(path, **kwargs))
        dataset = AudioDataset(path, write_index=write_index, **kwargs)
        if dataset.sr is None:
            most_common_sr = Counter(f['sr'] for f in dataset.files).most_common(1)[0][0]
            dataset._sr = most_common_sr
        dataset = dataset.configure(chunk_length=chunk_length or dataset.chunk_length or 1.)
        return cls(dataset)

    # general information

    @property
    def is_online(self) -> bool:
        return isinstance(self.dataset, OnlineAudioDataset)

    @property
    def sr(self) -> int:
        if self.is_online:
            return self.dataset.sr
        return int(self.dataset.metadata['sr'])

    @property
    def feature_names(self) -> List[str]:
        features = self.dataset.features
        names = list(features.keys()) if isinstance(features, dict) else [f.feature_name for f in features]
        return names

    @property
    def array_feature_names(self) -> List[str]:
        """features that can be loaded as arrays through output patterns (path-based features are metadata)."""
        hashed = set(self.dataset.feature_hash.keys())
        return [f for f in self.feature_names if f not in hashed or not self.is_categorical(f)]

    @property
    def categorical_feature_names(self) -> List[str]:
        """hashed features (e.g. path-based metadata), whose values index dataset items."""
        return [f for f in self.dataset.feature_hash.keys() if f != "original_path"]

    def is_categorical(self, feature: str) -> bool:
        return feature in self.dataset.feature_hash and feature != "original_path"

    def summary(self) -> Dict[str, Any]:
        files = self.dataset.files
        durations = [f['duration'] for f in files]
        summary = {
            'mode': "online" if self.is_online else "pre-processed",
            'path': self.dataset.path,
            'n_items': len(self.dataset),
            'chunk_length': self.dataset.chunk_length,
            'chunk_duration': self.dataset.chunk_length / self.sr if self.dataset.chunk_length else None,
            'sr': self.sr,
            'n_files': len(files),
            'total_duration': sum(durations),
            'file_durations': {'min': min(durations), 'median': statistics.median(durations), 'max': max(durations)} if files else None,
            'channels': dict(Counter(f['channels'] for f in files)),
            'features': self.feature_names,
            'categorical_features': self.categorical_feature_names,
            'partitions': self.list_partitions(),
        }
        if self.is_online:
            summary['batch_by'] = self.dataset.batch_by
            summary['file_sample_rates'] = dict(Counter(f['sr'] for f in files))
            summary['invalid_files'] = self.dataset.invalid_files
        else:
            metadata = self.dataset.metadata
            for k in ['preset', 'hop_length', 'waveform', 'skipped_files', 'format']:
                if k in metadata: summary[k] = metadata[k]
        return _jsonable(summary)

    def file_table(self) -> List[List[Any]]:
        """rows of (index, audio path, duration, sr, channels, items or metadata)."""
        rows = []
        for i, f in enumerate(self.dataset.files):
            extra = f.get('n_chunks') if not self.is_online else ", ".join("%s=%s"%(k, v) for k, v in f['metadata'].items())
            rows.append([i, f['audio_path'], round(f['duration'], 3), f['sr'], f['channels'], extra])
        return rows

    # items and files

    def item(self, index: int) -> Dict[str, Any]:
        """audio, features and metadata of a dataset item."""
        index = int(index) % len(self.dataset)
        fragment = self.dataset._fragment_from_index(index)
        audio = _to_numpy(fragment.get_audio("waveform")) if fragment.has_buffer("waveform") else None
        features = {}
        for name in self.feature_names:
            if fragment.has_buffer(name):
                try:
                    features[name] = _to_numpy(fragment.get(name))
                except Exception as e:
                    features[name] = "could not load (%s)"%e
        metadata = {k: v for k, v in fragment.get_metadata().items() if k != "output_type"}
        return {'index': index, 'audio': audio, 'sr': self.sr, 'features': features, 'metadata': metadata}

    def file_audio(self, file_idx: int, start: float = 0., length: float | None = None) -> Tuple[int, np.ndarray]:
        file_idx = int(file_idx)
        audio = self.dataset.load_audio(file_idx, start=float(start), length=None if not length else float(length))
        return self.sr, _to_numpy(audio)

    # statistics

    def _sample_indices(self, n_items: int | None, seed: int = 0) -> List[int]:
        indices = list(range(len(self.dataset)))
        if n_items is None or n_items >= len(indices):
            return indices
        return sorted(random.Random(seed).sample(indices, int(n_items)))

    def feature_values(self, feature: str, n_items: int | None = 100, seed: int = 0) -> np.ndarray:
        """values of a feature over n_items random items (None for the whole dataset), flattened. For online
        datasets, features are computed on loaded items."""
        values = []
        for i in self._sample_indices(n_items, seed):
            value = self.dataset.get(i, output_pattern=feature, transforms=None)
            values.append(_to_numpy(value).astype(np.float64).reshape(-1))
        return np.concatenate(values) if values else np.array([])

    def feature_stats(self, feature: str, n_items: int | None = 100, seed: int = 0) -> Dict[str, Any]:
        """statistics of a numeric feature (see feature_values), ignoring non-finite values."""
        values = self.feature_values(feature, n_items=n_items, seed=seed)
        finite = values[np.isfinite(values)]
        n_items_used = len(self._sample_indices(n_items, seed))
        stats = {'feature': feature, 'n_items': n_items_used, 'n_values': int(values.size),
                 'n_non_finite': int(values.size - finite.size)}
        if finite.size > 0:
            stats.update({
                'mean': float(finite.mean()), 'std': float(finite.std()), 'min': float(finite.min()), 'max': float(finite.max()),
                'percentiles': {str(p): float(np.percentile(finite, p)) for p in [1, 5, 25, 50, 75, 95, 99]}
            })
        return {'stats': stats, 'values': finite}

    def categorical_counts(self, feature: str) -> Dict[str, int]:
        """number of items for each value of a hashed feature (see partition_report for partitions)."""
        counts = self._partition_counts(self.dataset, feature)
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def _partition_counts(self, dataset, feature) -> Dict[str, int]:
        feature_hash = dataset.feature_hash[feature]
        if isinstance(dataset, OnlineAudioDataset):
            # online hashes index files : count items of these files
            if dataset.chunk_length is None:
                return {str(k): len(v) for k, v in feature_hash.items()}
            return {str(k): len(dataset.item_indices_for_files(v)) for k, v in feature_hash.items()}
        current_keys = dataset._current_keys()
        counts = {}
        for value, keys in feature_hash.items():
            keys = set(map(dataset._key_to_str, keys))
            counts[str(value)] = len(keys) if current_keys is None else len(keys & current_keys)
        return counts

    # augmentations

    @staticmethod
    def available_transforms() -> List[str]:
        """names of transform configurations (see acids_dataset/configs/transforms)."""
        from .. import TRANSFORM_GIN_PATH
        return sorted(os.path.splitext(f)[0] for f in os.listdir(TRANSFORM_GIN_PATH) if f.endswith(".gin"))

    def get_transform(self, name: str):
        """instantiates a transform from its configuration, at the sampling rate of the dataset."""
        if name not in self._transform_cache:
            from ..utils import transform_from_gin_config
            from ..presets import preset_env, quiet_gin_search
            with preset_env(None, values={"SAMPLE_RATE": self.sr}), quiet_gin_search():
                self._transform_cache[name] = transform_from_gin_config(name, add_args=((), {"sr": self.sr}))
        return self._transform_cache[name]

    def augment(self, audio: np.ndarray | torch.Tensor, transforms: List[str], force: bool = True) -> Tuple[np.ndarray, List[str]]:
        """applies transforms (by name) to audio of shape (channels, samples). If force, transforms are always applied ;
        otherwise, they are applied randomly with their probability. Returns the augmented audio, and errors of
        transforms that could not be applied."""
        x = torch.as_tensor(np.asarray(audio, dtype=np.float32))[None]
        errors = []
        for name in transforms:
            try:
                transform = self.get_transform(name)
                try:
                    y = transform(x, _force_transform=force)
                except TypeError:
                    # some transforms (e.g. Crop) do not have random behaviours
                    y = transform(x)
                y = y[0] if isinstance(y, list) else y
                x = torch.as_tensor(np.asarray(_to_numpy(y), dtype=np.float32))
            except Exception as e:
                errors.append("%s : %s"%(name, str(e).splitlines()[0] if str(e) else type(e).__name__))
        return _to_numpy(x[0]), errors

    # health

    def health(self, max_files: int | None = None) -> Dict[str, Any]:
        """health report of the files of the dataset (see acids_dataset.health.dataset_health)."""
        from ..health import dataset_health
        return _jsonable(dataset_health(self.dataset, max_files=max_files))

    # partitions

    def list_partitions(self) -> List[str]:
        if self.is_online:
            folder, ext = self.dataset.partition_dir, ".json"
        else:
            folder, ext = Path(self.dataset.path) / "partitions", ".txt"
        if not Path(folder).is_dir():
            return []
        return sorted(os.path.splitext(f)[0] for f in os.listdir(folder) if f.endswith(ext))

    def load_partitions(self, name: str) -> Dict[str, AudioDataset]:
        return self.dataset.load_partition(name)

    def _partition_files(self, dataset) -> set:
        if isinstance(dataset, OnlineAudioDataset):
            return {dataset._files[i]['audio_path'] for i in dataset.file_indices}
        keys = dataset._current_keys()
        files = set()
        for audio_path, path_keys in dataset.feature_hash['original_path'].items():
            if keys is None or len(set(map(dataset._key_to_str, path_keys)) & keys) > 0:
                files.add(audio_path)
        return files

    def partition_report(self, name_or_partitions: str | Dict[str, AudioDataset]) -> Dict[str, Any]:
        """sizes of partitions (given by name or as a dict of datasets), distribution of categorical features, and 
        files shared between partitions (that may leak data between training and validation)."""
        partitions = self.load_partitions(name_or_partitions) if isinstance(name_or_partitions, str) else name_or_partitions
        files = {p: self._partition_files(d) for p, d in partitions.items()}
        sizes = {p: {'n_items': len(d), 'ratio': len(d) / max(1, len(self.dataset)), 'n_files': len(files[p])} for p, d in partitions.items()}
        shared = {}
        names = list(partitions.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                common = files[names[i]] & files[names[j]]
                if common:
                    shared["%s / %s"%(names[i], names[j])] = sorted(common)
        distributions = {f: {p: self._partition_counts(d, f) for p, d in partitions.items()} for f in self.categorical_feature_names}
        return {'sizes': sizes, 'shared_files': shared, 'distributions': distributions}

    @property
    def split_features(self) -> List[str]:
        """features that can be used to group items in partitions."""
        return ["original_path"] + self.categorical_feature_names

    @staticmethod
    def parse_ratios(ratios: str | Dict[str, float]) -> Dict[str, float]:
        """parses partition ratios given as "train: 0.8, valid: 0.2" (or "train=0.8, valid=0.2")."""
        if isinstance(ratios, dict):
            parsed = {str(k): float(v) for k, v in ratios.items()}
        else:
            parsed = {}
            for entry in filter(None, map(str.strip, ratios.replace(";", ",").split(","))):
                if ":" in entry: name, value = entry.split(":", 1)
                elif "=" in entry: name, value = entry.split("=", 1)
                else: raise ValueError("could not parse partition %s ; partitions must be given as name: ratio"%entry)
                name = name.strip()
                if not name: raise ValueError("partition names cannot be empty")
                if name in parsed: raise ValueError("partition %s is given twice"%name)
                parsed[name] = float(value)
        if len(parsed) == 0:
            raise ValueError("no partition given")
        if any(v <= 0 for v in parsed.values()):
            raise ValueError("partition ratios must be positive")
        if sum(parsed.values()) > 1 + 1e-6:
            raise ValueError("partition ratios sum to %.3f, that exceeds 1"%sum(parsed.values()))
        return parsed

    def create_partition(self, ratios: str | Dict[str, float], features: List[str] | None = None, balance_cardinality: bool = False, 
                         tolerance: float = 0.1, seed: int | None = 0, use_partition_feature: bool = False, 
                         name: str | None = None, overwrite: bool = False) -> Dict[str, Any]:
        """creates partitions with dataset.split, and writes them with the given name (if not None). Returns the 
        report of the partitions (see partition_report)."""
        ratios = self.parse_ratios(ratios)
        features = [f for f in (features or []) if f]
        for f in features:
            if f not in self.split_features:
                raise ValueError("feature %s cannot be used for partitions ; available features : %s"%(f, self.split_features))
        if name is not None:
            name = name.strip()
            if not name: raise ValueError("partition name cannot be empty")
            if os.path.basename(name) != name or name.startswith("."): raise ValueError("invalid partition name %s"%name)
            if not overwrite and name in self.list_partitions():
                raise FileExistsError("partition %s already exists"%name)
        partitions = self.dataset.split(ratios, features=features or None, balance_cardinality=balance_cardinality, tolerance=tolerance, 
                                        seed=seed, use_meta_if_available=use_partition_feature, write=name or None)
        return self.partition_report(partitions)


__all__ = ['DatasetBrowser']
