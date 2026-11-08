import gin
import torch
import numpy as np
from types import EllipsisType
import random
from math import ceil
import os
from absl import logging
import random, math, itertools
from collections import deque
import copy
from typing import Optional, List , Dict, Iterable
from ..utils import checklist, pad, PadMode
from .. import get_writer_class_from_path, get_metadata_from_path, get_fragment_class
from .base import AudioDataset, TransformType, chunk_length_to_samples


@gin.configurable(module="datasets")
class PreprocessedAudioDataset(AudioDataset):
    """AudioDataset loading fragments from a pre-processed dataset folder (see preprocess_dataset). Pre-processed features
    are read from the database, and describe the original audio ; features_after_transforms only applies to batch features
    (see get_loader)."""
    _default_partition_name = "partitions"

    @classmethod
    def _init_from_preset(cls, preset, preset_overrides, *args, **kwargs):
        from ..presets import preset_env, query_preset_value
        with preset_env(preset, overrides=preset_overrides):
            dataset = type.__call__(cls, *args, **kwargs)
            # pre-processed data is fixed : check that it is consistent with the preset
            metadata = dataset.metadata
            preset_sr = query_preset_value("SAMPLE_RATE")
            preset_chunk_length = query_preset_value("CHUNK_LENGTH")
            if preset_sr is not None and preset_sr != metadata.get('sr'):
                logging.warning("dataset %s was pre-processed with sr=%s, but preset %s has SAMPLE_RATE=%s"%(dataset.path, metadata.get('sr'), preset, preset_sr))
            if preset_chunk_length is not None and chunk_length_to_samples(preset_chunk_length, metadata['sr']) != dataset.chunk_length:
                logging.warning("dataset %s was pre-processed with chunk_length=%s samples, but preset %s has CHUNK_LENGTH=%s"%(dataset.path, dataset.chunk_length, preset, preset_chunk_length))
        return dataset
    def __init__(self,
                 path: str | None = None,
                 transforms: TransformType = None, 
                 output_pattern: str = 'waveform',
                 required_fields: Iterable[str] = [],
                 output_type: str = "torch",
                 channels: int = 1, 
                 lazy_paths: bool = False,
                 subindices: Iterable[int] | Iterable[bytes] | None = None,
                 parent = None,
                 max_samples: int | None = None,
                 db_path: str | None = None,
                 features_after_transforms: bool = True,
                 **kwargs) -> None:
        # db_path kept for retro-compatibility
        path = path or db_path
        from .online import OnlineAudioDataset
        self._check_kwargs(kwargs, OnlineAudioDataset)
        self._db_path = path
        if lazy_paths:
            raise NotImplementedError()
        self._init_common(transforms=transforms, output_pattern=output_pattern, output_type=output_type, 
                          channels=channels, subindices=subindices, max_samples=max_samples,
                          features_after_transforms=features_after_transforms)
        self._loader = get_writer_class_from_path(path).loader(self._db_path, output_type=output_type) 
        self._metadata = get_metadata_from_path(path)
        self._required_fields = required_fields
        # file information, built when needed
        self._files = None
        self._chunk_positions = {}
        if (self._required_fields):
            self._subindices = self._parse_required_fields(self._subindices, self._required_fields)
        if parent:
            self.parent = parent
        torch.utils.data.Dataset.__init__(self)

    def __len__(self):
        if self._subindices is None:
            return len(self._loader)
        else:
            return len(self._subindices)

    def _fragment_from_index(self, index):
        if (self._subindices is not None) and isinstance(index, int):
            index = self._subindices[index]
        elif (self._subindices is not None) and isinstance(index, (str, bytes)):
            #TODO think about this : allow or not? 
            pass
        return self._loader[index]

    def configure(self, chunk_length: int | float | None = None, features_after_transforms: bool | None = None, **kwargs) -> "PreprocessedAudioDataset":
        if chunk_length is not None and chunk_length_to_samples(chunk_length, self._metadata['sr']) != self.chunk_length:
            raise ValueError("chunk_length of pre-processed datasets is fixed (here, %d samples)"%self.chunk_length)
        return super().configure(features_after_transforms=features_after_transforms, **kwargs)

    @property
    def chunk_length(self) -> int:
        return chunk_length_to_samples(float(self._metadata['chunk_length']), int(self._metadata['sr']))

    @property
    def path(self):
        return str(self._db_path)

    # file access : audio of original files is rebuilt from stored chunks

    @property
    def files(self) -> List[Dict]:
        """original files of the dataset, with audio_path, duration, sr, channels, n_frames, n_chunks and keys (chunk 
        keys in the database). File indices refer to the whole dataset (also for partitions)."""
        if self._files is None:
            sr = int(self._metadata['sr'])
            files = []
            for audio_path, keys in self._loader.feature_hash['original_path'].items():
                if len(keys) == 0: continue
                keys = [self._key_to_str(k) for k in keys]
                fragment = self._loader[keys[0]]
                duration = float(fragment.get_metadata().get('file_length', 0.))
                channels = fragment.get_buffer('waveform').shape[0] if fragment.has_buffer('waveform') else self._metadata.get('channels')
                files.append({'audio_path': audio_path, 'duration': duration, 'sr': sr, 'channels': channels, 
                              'n_frames': int(round(duration * sr)), 'n_chunks': len(keys), 'keys': keys})
            self._files = files
        return copy.deepcopy(self._files)

    def _file_chunk_positions(self, file_idx: int) -> List[tuple]:
        """returns (start sample, key) of the chunks of a file, sorted by position."""
        if file_idx not in self._chunk_positions:
            file = self.files[file_idx]
            positions = []
            for key in file['keys']:
                start_pos = float(self._loader[key].get_metadata().get('start_pos', 0.))
                positions.append((int(round(start_pos * file['sr'])), key))
            self._chunk_positions[file_idx] = sorted(positions)
        return self._chunk_positions[file_idx]

    def load_audio(self, file_idx: int, start: float = 0., length: int | float | None = None, pad_mode: PadMode | str | None = None) -> torch.Tensor:
        """loads audio of a given file, rebuilt from the chunks stored in the database (at the sampling rate of the 
        dataset). Parts of the file that were not stored (e.g. discarded chunks) are filled with zeros, and audio 
        requested beyond the end of the file is padded with pad_mode (zero by default).

        Args:
            file_idx (int): file index (in self.files).
            start (float, optional): start position (seconds). Defaults to 0.
            length (int | float | None, optional): length, in samples (int) or seconds (float). None for the whole file. Defaults to None.
            pad_mode (PadMode | str | None, optional): padding mode beyond the end of the file. Defaults to None (zero).
        """
        if not self._metadata.get('waveform', True):
            raise ValueError("dataset %s was pre-processed without waveform : audio cannot be loaded"%self.path)
        file = self.files[file_idx]
        sr, n_file = file['sr'], file['n_frames']
        start_smp = int(round(start * sr))
        n = max(0, n_file - start_smp) if length is None else chunk_length_to_samples(length, sr)
        out = torch.zeros(file['channels'], n)
        # samples of the file that are available
        end_smp = min(start_smp + n, n_file)
        for chunk_start, key in self._file_chunk_positions(file_idx):
            if chunk_start >= end_smp: break
            if chunk_start + self.chunk_length <= start_smp: continue
            chunk = torch.as_tensor(self._loader[key].get_audio('waveform')).float()
            chunk = chunk.reshape(-1, chunk.shape[-1])
            # chunks may be padded beyond the end of the file
            lo, hi = max(chunk_start, start_smp), min(chunk_start + chunk.shape[-1], end_smp)
            if hi > lo:
                out[:, lo - start_smp:hi - start_smp] = chunk[:, lo - chunk_start:hi - chunk_start]
        n_valid = max(0, end_smp - start_smp)
        if n_valid < n:
            out = pad(out[..., :n_valid], n, pad_mode or PadMode.ZERO)
        return out

    def get_fragment(self, file_idx: int, start: float = 0., length: int | float | None = None, features = None, **kwargs):
        """loads audio of a given file (see load_audio) as a fragment, with features computed on the loaded audio 
        (None for the features of the dataset)."""
        file = self.files[file_idx]
        wav = self.load_audio(file_idx, start=start, length=length, **kwargs)
        features = list(self.features.values()) if features is None else features
        fragment_class = get_fragment_class(self._metadata.get('fragment_class', 'AcidsFragment'))
        return self._build_fragment(fragment_class, wav, file['sr'], file['audio_path'], start, file['duration'], features, 
                                    output_type=self._output_type)

    @property 
    def metadata(self):
        return copy.copy(self._metadata)

    @property
    def loader(self): 
        return self._loader

    @property
    def features(self) :
        return self._loader.features

    @property
    def feature_hash(self): 
        return self._loader.feature_hash

    @property
    def keys(self) -> List[str]:
        return list(self._loader.iter_fragment_keys(as_bytes=False))

    def _parse_required_fields(self, subindices, required_fields):
        if subindices is None: 
            subindices = range(len(self))
        subindices = set(subindices)
        if not self._loader.feature_status: 
            return None
        for f in required_fields: 
            assert f in self._loader.feature_status, f"{f} not found in dataset"
            status = self._loader.feature_status[f]
            if status.count() >= len(self): 
                continue
            subindices.intersection_update(status.nonzero())
        return list(subindices)
           
    def index_for_features(self, **kwargs) -> Dict[tuple, set]: 
        """returns {(value_1, ..., value_n): set of keys} for given feature values (e.g. index_for_features(instrument=["piano", "guitar"], idx=0)), 
        with an entry for each combination of values."""
        feature_hash = self._loader.feature_hash
        current_keys = self._current_keys()
        # None stands for every key of the dataset
        idx_dict = {(): current_keys}
        for feature_name, feature_value in kwargs.items(): 
            assert feature_name in feature_hash, "feature %s not present in dataset"%feature_name
            new_idx_dict = {}
            for v in checklist(feature_value): 
                # regexp values are strings
                hash_tmp = feature_hash[feature_name].get(v, feature_hash[feature_name].get(str(v)))
                if not hash_tmp: raise IndexError("could not find value %s for feature %s"%(v, feature_name))
                hash_tmp = set(map(self._key_to_str, hash_tmp))
                for kk, vv in idx_dict.items(): 
                    new_idx_dict[kk + (v,)] = hash_tmp if vv is None else vv.intersection(hash_tmp)
            idx_dict = new_idx_dict
        return idx_dict

    def query_from_features(self, n: int | EllipsisType, _random: bool = True, **kwargs):
        assert isinstance(n, (int, EllipsisType)), "n must be either an int or Ellipsis"
        idx_dict = self.index_for_features(**kwargs)
        data_dict = {}
        for k, v in idx_dict.items(): 
            idx_dict[k] = sorted(v)
            if n != Ellipsis and len(idx_dict[k]) > 0: 
                if _random: 
                    idx_dict[k] = random.choices(idx_dict[k], k=n)
                else:
                    idx_dict[k] = idx_dict[k][:n]
            data_dict[k] = [self[i] for i in idx_dict[k]]
        
        return data_dict 


    # split functions

    def get_subdataset(self, subindices, check=False):
        if check: 
            for s in subindices: 
                if s not in self._loader: 
                    raise IndexError('key %s not found in dataset'%s)
        subdataset = copy.copy(self)
        subdataset._subindices = subindices
        subdataset.parent = self
        return subdataset

    def _split_without_feature(self, _rng=random, **kwargs):
        ratios = {k: float(v) for k, v in kwargs.items()}
        n_items = len(self._loader) if not self._subindices else len(self._subindices)
        idx_perm = _rng.sample(range(n_items), k=n_items)
        current_idx = 0
        subindices = {}
        for i, k in enumerate(ratios.keys()):
            part_len = int(ratios[k] * n_items)
            subindices[k] = idx_perm[current_idx:current_idx+part_len]
            current_idx += part_len
        if self._subindices is not None: 
            for k in subindices:
                subindices[k] = [self._subindices[i] for i in subindices[k]]
        subindices = {k: list(map(lambda x: self._loader._keys[x], subindices[k])) for k in subindices}
        return {k: self.get_subdataset(subindices[k]) for k in ratios.keys()}           

    def item_labels(self, features, n_bins: int = 10):
        from ..sampling import bin_values
        keys = self._subindices if self._subindices is not None else self._loader._keys
        keys = [self._key_to_str(k) for k in keys]
        feature_hash = self._loader.feature_hash
        per_feature = []
        for f in checklist(features):
            if f in feature_hash:
                key_values = {}
                for value, value_keys in feature_hash[f].items():
                    for k in value_keys:
                        key_values[self._key_to_str(k)] = value
                per_feature.append([key_values.get(k) for k in keys])
            elif f in self.features:
                per_feature.append(bin_values([self._reduce_feature_value(self._loader[k].get(f)) for k in keys], n_bins=n_bins))
            else:
                raise ValueError("feature %s not present in dataset ; available features : %s"%(f, list(feature_hash.keys()) + list(self.features.keys())))
        return self._combine_labels(per_feature)

    def _key_to_str(self, key) -> str:
        if isinstance(key, int):
            key = self._loader._keys[key]
        if isinstance(key, bytes):
            key = self._loader.keygen.to_str(key)
        return key

    def _current_keys(self) -> set | None:
        """returns the keys (as str) of the dataset if it is restricted by subindices, None otherwise."""
        if self._subindices is None:
            return None
        return set(map(self._key_to_str, self._subindices))

    def _sort_split_features(self, features) -> tuple:
        """separates hashed (categorical) features from other stored (continuous) features."""
        feature_hash = self._loader.feature_hash
        categorical, continuous = [], []
        for f in checklist(features):
            if f in feature_hash: categorical.append(f)
            elif f in self.features: continuous.append(f)
            else: raise ValueError("feature %s not present in dataset ; available features : %s"%(f, list(feature_hash.keys()) + list(self.features.keys())))
        if len(continuous) > 1:
            raise ValueError("partitions can only be stratified over one continuous feature, got %s"%continuous)
        return categorical, (continuous[0] if continuous else None)

    def _split_stratified(self, partitions, partition_hash, feature_name, n_bins=10, rng=random):
        """assigns groups of keys to partitions, stratified over the mean value of a continuous feature."""
        labels, weights, values = [], [], []
        n_missing = 0
        for label, keys in partition_hash.items():
            item_values = [self._reduce_feature_value(self._loader[k].get(feature_name)) for k in keys]
            item_values = [v for v in item_values if np.isfinite(v)]
            if not item_values:
                n_missing += len(keys)
                continue
            labels.append(label)
            weights.append(float(len(keys)))
            values.append(float(np.mean(item_values)))
        if n_missing > 0:
            logging.warning("%d items have no finite value for feature %s, and are not partitioned"%(n_missing, feature_name))
        assignment = self._assign_stratified(labels, weights, values, partitions, n_bins=n_bins, rng=rng)
        subindices = {p: sorted(sum([list(partition_hash[l]) for l in assignment[p]], [])) for p in partitions.keys()}
        return {p: self.get_subdataset(list(map(self._loader.keygen.from_str, subindices[p]))) for p in partitions.keys()}

    def _split_with_features(self, partitions, features, tolerance=0.1, balance_cardinality = False, rng=random, n_bins=10):
        features, continuous = self._sort_split_features(features)
        if continuous is not None and len(features) == 0:
            # items are stratified over the continuous feature
            keys = self._current_keys()
            keys = sorted(keys) if keys is not None else [self._key_to_str(k) for k in self._loader._keys]
            return self._split_stratified(partitions, {(k,): {k} for k in keys}, continuous, n_bins=n_bins, rng=rng)
        # make partition hash : {(value_1, ..., value_n): set of keys (as str)}
        feature_hash = self._loader.feature_hash
        for f in features:
            assert f in feature_hash, f"feature {f} not present in dataset"
        current_keys = self._current_keys()
        partition_hash = {}
        for current_hash in itertools.product(*[feature_hash[f].keys() for f in features]):
            subset = None
            for i, f in enumerate(features):
                keys = set(map(self._key_to_str, feature_hash[f][current_hash[i]]))
                subset = keys if subset is None else subset.intersection(keys)
            if current_keys is not None:
                subset.intersection_update(current_keys)
            if len(subset) > 0:
                partition_hash[current_hash] = subset

        if continuous is not None:
            # groups of categorical features (e.g. original_path) are stratified over the continuous feature
            return self._split_stratified(partitions, partition_hash, continuous, n_bins=n_bins, rng=rng)
        if balance_cardinality: 
            return self._split_with_features_balanced(partitions, partition_hash, features, tolerance=tolerance, rng=rng)
        else:
            return self._split_with_features_unbalanced(partitions, partition_hash, features, tolerance=tolerance, rng=rng)

    def _split_with_features_unbalanced(self, partitions, partition_hash, features, tolerance=0.1, rng=random):
        # feature values are distributed according to partition ratios
        labels = list(partition_hash.keys())
        rng.shuffle(labels)
        assignment = self._assign_by_ratio(labels, [1.] * len(labels), partitions)
        subindices = {p: [] for p in partitions.keys()}
        for p, current_labels in assignment.items():
            for c in current_labels:
                # keys are sorted, such that partitions are identical across processes
                subindices[p].extend(map(self._loader.keygen.from_str, sorted(partition_hash[c])))
        return {p: self.get_subdataset(subindices[p]) for p in partitions.keys()}

    def _split_with_features_balanced(self, partitions, partition_hash, features, tolerance=0.1, rng=random):
        # feature values are assigned such that partition sizes follow given ratios
        labels = rng.sample(list(partition_hash.keys()), k=len(partition_hash))
        weights = [len(partition_hash[l]) for l in labels]
        assignment = self._assign_balanced(labels, weights, partitions, tolerance=tolerance)
        subindices = {p: sorted(sum([list(partition_hash[l]) for l in labels], [])) for p, labels in assignment.items()}
        return {p: self.get_subdataset(list(map(self._loader.keygen.from_str, subindices[p]))) for p in partitions.keys()}

    def _split_from_feature(self, partition_key,  **kwargs): 
        assert partition_key in self.feature_hash, "key %s not found in feature_hash"%(partition_key)
        partition_names = list(kwargs.keys())
        for p in partition_names: 
            assert p in self.feature_hash[partition_key], "partition %s not found in %s"%(p, partition_key)
        return {k: self.get_subdataset(self.feature_hash[partition_key][k]) for k in partition_names}

    def _write_partition(self, partitions, write: str | bool):
        partition_dir = os.path.join(self._db_path, "partitions")
        if write is True:
            write = type(self)._default_partition_name
        if os.path.splitext(write)[1] != ".txt": write += ".txt"
        os.makedirs(partition_dir, exist_ok=True)
        partition_file = os.path.join(partition_dir, write)
        with open(partition_file, "wb+") as f:
            for p, v in partitions.items(): 
                f.write(f"{p};".encode('utf-8'))
                f.write(b",".join(v._subindices))
                f.write(("\n").encode('utf-8'))

    def has_partition(self, name: str):
        if os.path.splitext(name)[1] != ".txt": name += ".txt"
        partition_path = os.path.join(self._db_path, "partitions", name)
        return os.path.exists(partition_path)

    def split(self, 
              partitions, 
              features=None, 
              use_meta_if_available: str | bool | None = True,
              load_if_available: bool | str | None = False,
              balance_cardinality: bool = False, 
              tolerance: float = 0.1, 
              write: str | bool | None = None,
              seed: int | None = None,
              n_bins: int = 10): 
        """splits the dataset in partitions. If seed is given, partitions are reproducible (e.g. identical across 
        processes of distributed training) ; otherwise, they are drawn with the global random state. Features can be 
        hashed (categorical) features, whose values are kept in a single partition, and at most one other (continuous) 
        feature, over which partitions are stratified with n_bins histogram (quantile) bins."""
        assert not self.is_partition, "dataset is already a partition of an existing dataset."
        rng = random.Random(seed) if seed is not None else random

        # first, try to load if required
        if load_if_available: 
            if (load_if_available is True): load_if_available = self._default_partition_name
            try: 
                partitions = self.load_partition(load_if_available)
                if features is not None: logging.warning("found partition to load, but features was not None; feature-based partitioning ignored")
                return partitions
            except FileNotFoundError: 
                pass

        # then, try to use metadata 
        if use_meta_if_available: 
            try:
                if (use_meta_if_available is True): use_meta_if_available = "partition"
                if use_meta_if_available in self.feature_hash: 
                    partitions = self._split_from_feature(use_meta_if_available, **partitions)
                    if features is not None: logging.warning("found partition to load, but features was not None; feature-based partitioning ignored")
                    return partitions
            except Exception as e: 
                logging.warning('Could not find partition using feature %s. Creating new partition'%use_meta_if_available)
                
        # fetch by features
        if features is None:
            partitions = self._split_without_feature(_rng=rng, **partitions)
        else:
            partitions = self._split_with_features(partitions, features, tolerance = tolerance, balance_cardinality=balance_cardinality, rng=rng, n_bins=n_bins)
        if write:            self._write_partition(partitions, write)
        return partitions

    def load_partition(self, name: str | None = None, check: bool = False):
        if name is None: name = self._default_partition_name
        if os.path.splitext(name)[1] != ".txt": name += ".txt"
        partition_path = os.path.join(self._db_path, "partitions", name)
        if not os.path.exists(partition_path):
            raise FileNotFoundError("could not load partition %s"%name)
        partitions = {}
        with open(partition_path, 'rb') as f:
            for l in f.readlines():
                name, subindices = l.split(b';')
                subindices = subindices.replace(b'\n', b'')
                partitions[name.decode('utf-8')] = self.get_subdataset(subindices.split(b','), check)
        return partitions


__all__ = ['PreprocessedAudioDataset']
