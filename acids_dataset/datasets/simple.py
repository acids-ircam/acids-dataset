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
from ..utils import checklist
from .. import transforms, get_writer_class_from_path, get_metadata_from_path
from .utils import _outs_from_pattern, _transform_outputs

TransformType = Optional[transforms.Transform | List[transforms.Transform] | Dict[str, transforms.Transform]]


def _parse_transforms_with_pattern(transform, pattern):
    #TODO better pattern / transform check
    return transform

class AudioDataset(torch.utils.data.Dataset):
    _default_partition_name = "partitions"
    def __init__(self,
                 db_path: str,
                 transforms: TransformType = None, 
                 output_pattern: str = 'waveform',
                 required_fields: Iterable[str] = [],
                 output_type: str = "torch",
                 channels: int = 1, 
                 lazy_import: bool = False, 
                 lazy_paths: bool = False,
                 subindices: Iterable[int] | Iterable[bytes] | None = None,
                 parent = None,
                 max_samples: int | None = None,
                 **kwargs) -> None:
        self._db_path = db_path
        if lazy_import or lazy_paths:
            raise NotImplementedError()
        self._loader = get_writer_class_from_path(db_path).loader(self._db_path, output_type=output_type) 
        self._metadata = get_metadata_from_path(db_path)
        self._output_pattern = output_pattern
        self._transforms = _parse_transforms_with_pattern(transforms, self._output_pattern)
        self._n_channels = channels
        self._subindices = subindices
        self._required_fields = required_fields
        if (self._required_fields):
            self._subindices = self._parse_required_fields(self._subindices, self._required_fields)
        self._parent = None
        self._partitions = {}
        # self._index_mapper = lambda x: x
        self._max_samples = max_samples
        if parent:
            self.parent = parent
        super(AudioDataset, self).__init__()

    def __getitem__(self, index):
        return self.get(index)
    
    def __len__(self):
        if self._subindices is None:
            return len(self._loader)
        else:
            return len(self._subindices)

    def get_sampler(self, replacement=False, max_samples=None): 
        max_samples = max_samples or self._max_samples or len(self)
        return torch.utils.data.RandomSampler(self, replacement=replacement, num_samples = max_samples)

    @property
    def path(self):
        return str(self._db_path)

    @property 
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, obj):
        assert issubclass(type(obj), type(self)), "parent of a dataset must be a subclass of %s"%(type(self))

    @property 
    def metadata(self):
        return copy.copy(self._metadata)

    @property
    def loader(self): 
        return self._loader

    @property
    def is_partition(self):
        return self._parent is not None

    @property
    def output_pattern(self):
        return self._output_pattern
    
    @output_pattern.setter
    def output_pattern(self, pattern):
        assert isinstance(pattern, str)
        self._output_pattern = pattern
        self._transforms = _parse_transforms_with_pattern(self._transforms, self._output_pattern)

    @property
    def transforms(self):
        return self._transforms

    @property
    def features(self) :
        return self._loader.features

    @property
    def feature_hash(self): 
        return self._loader.feature_hash

    @transforms.setter
    def transforms(self, transforms):
        self._transforms = _parse_transforms_with_pattern(transforms, self._output_pattern)

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
           
    # index functions
    def get(self, index, output_pattern=None, transforms=None):
        output_pattern = output_pattern or self.output_pattern
        transforms = transforms or self.transforms
        if (self._subindices is not None) and isinstance(index, int):
            index = self._subindices[index]
        elif (self._subindices is not None) and isinstance(index, (str, bytes)):
            #TODO think about this : allow or not? 
            pass
        fg = self._loader[index]
        outs = _outs_from_pattern(fg, output_pattern)
        outs = _transform_outputs(outs, self.transforms, self._n_channels)
        return outs
    
    def index_for_features(self, **kwargs): 
        idx_dict = {}
        for feature_name, feature_value in kwargs.items(): 
            assert feature_name in self._loader.feature_hash
            for v in checklist(feature_value): 
                hash_tmp = self._loader.feature_hash[feature_name].get(v)
                if hash_tmp is None or len(hash_tmp) == 0: raise IndexError("could not find value %d for feature %s"%(v, feature_name))
                if len(idx_dict) == 0: 
                    idx_dict[(v,)] = set(hash_tmp)
                else:
                    new_idx_dict = {}
                    for kk, vv in idx_dict.items(): 
                        new_idx_dict[kk + (v,)] = vv.intersection(hash_tmp)
                    idx_dict = new_idx_dict
        return idx_dict

    def query_from_features(self, n: int | EllipsisType, _random: bool = True, **kwargs):
        assert isinstance(n, (int, EllipsisType)), "n must be either an int or Ellipsis"
        idx_dict = self.index_for_features(**kwargs)
        data_dict = {}
        for k, v in idx_dict.items(): 
            idx_dict[k] = list(v)
            if n != Ellipsis: 
                if _random: 
                    idx_dict[k] = random.choices(idx_dict[k], k=n)
                else:
                    idx_dict[k] = v[:n]
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

    def _split_without_feature(self, **kwargs):
        ratios = {k: float(v) for k, v in kwargs.items()}
        n_items = len(self._loader) if not self._subindices else len(self._subindices)
        idx_perm = random.sample(range(n_items), k=n_items)
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

    def _split_with_features(self, partitions, features, tolerance=0.1, balance_cardinality = False):
        # make partition hash
        for f in features:
            assert f in self._loader.feature_hash, f"feature {f} not present in dataset"
        if len(features) == 1:
            partition_hash = {(k,): v for k, v in self._loader.feature_hash[features[0]].items()}
        else:
            partition_hash = {}
            for current_hash in itertools.product(*[self._loader.feature_hash[f] for f in features]):
                subset = None
                for i, f in enumerate(features):
                    if subset is None: 
                        subset = set(self._loader.feature_hash[f][current_hash[i]])
                    else:
                        subset.intersection_update(self._loader.feature_hash[f])
                partition_hash[current_hash] = subset

        if balance_cardinality: 
            return self._split_with_features_balanced(partitions, partition_hash, features, tolerance=tolerance)
        else:
            return self._split_with_features_unbalanced(partitions, partition_hash, features, tolerance=tolerance)

    def _split_with_features_unbalanced(self, partitions, partition_hash, features, tolerance=0.1):
        labels = list(partition_hash.keys())
        n_items = len(labels)
        subindices = {p: [] for p in partitions.keys()}
        random.shuffle(labels)
        current_idx = 0
        for i, k in enumerate(partitions.keys()):
            part_len = int(partitions[k] * n_items)
            current_labels = labels[current_idx:current_idx+part_len]
            for c in current_labels:
                if self._subindices:
                    idx_to_add = set(partition_hash[c]).intersection(self._subindices)
                else:
                    idx_to_add = partition_hash[c]
                subindices[k].extend(map(self._loader.keygen.from_str, idx_to_add))
            current_idx += part_len

        partitions = {p: self.get_subdataset(subindices[p]) for p in partitions.keys()}
        return partitions

    def _split_with_features_balanced(self, partitions, partition_hash, features, tolerance=0.1):
        sorted_keys = list(sorted(partition_hash.keys(), key=lambda x: len(partition_hash[x]),)) 
        # make partitions
        n_examples = len(self) if not self._subindices else len(self._subindices)
        pmap = {p: {'current_set': list(), 'target_len': int(partitions[p] * n_examples), 'tolerance': tolerance * n_examples * partitions[p], 'full': False} for p in partitions.keys()}
        unset = []
        part_names = deque(pmap.keys());
        # f = open('oops.txt', 'w+')
        for label in sorted_keys: 
            current_set = partition_hash[label]
            if self._subindices:
                current_set = list(set(current_set).intersection(self._subindices))
            current_set_distributed = False
            if set(pmap[p]['full'] for p in part_names) != {True}:
                for p in part_names:
                    if pmap[p]['full']: continue
                    if ((len(pmap[p]['current_set']) + len(current_set)) <  pmap[p]['target_len'] + pmap[p]['tolerance']):
                        # f.write(f"{p}, {abs((len(pmap[p]['current_set']) + len(current_set)) -  pmap[p]['target_len'])}, {pmap[p]['tolerance']}\n")
                        pmap[p]['current_set'].extend(map(self._loader._keygen.from_str, current_set))
                        current_set_distributed = True
                        break
                    else: 
                        pmap['full'] = True
            if not current_set_distributed:
                unset.append(current_set)
            part_names.rotate(1)
        partitions = {p: self.get_subdataset(pmap[p]['current_set']) for p in part_names}
        return partitions

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
              write: str | bool | None = None): 
        assert not self.is_partition, "dataset is already a partition of an existing dataset."

        # first, try to load if required
        if load_if_available: 
            if (load_if_available is True): load_if_available = type(self._default_partition_name)
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
            partitions = self._split_without_feature(**partitions)
        else:
            partitions = self._split_with_features(partitions, features, tolerance = tolerance, balance_cardinality=balance_cardinality)
        if write:            self._write_partition(partitions, write)
        return partitions

    def load_partition(self, name: str | None, check: bool = False):
        if name is None: name = type(self._default_partition_name)
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
    

                 

            

