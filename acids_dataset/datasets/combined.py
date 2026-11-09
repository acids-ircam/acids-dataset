from typing import List, Dict, Any
from pathlib import Path
import bisect
import math
import copy
from absl import logging
import torch
from functools import reduce
from .base import AudioDataset, TransformType
from ..utils import checklist
from .utils import is_multi_index, multi_index_to_list, collate_items

def cumsum(l):
    return [sum(l[:i+1]) for i in range(len(l))]


class CombinedAudioDataset(AudioDataset):
    """AudioDataset combining several datasets (pre-processed or online), with optional sampling weights. Items, files,
    partitions and loaders are those of the combined datasets, in order. Datasets can be given as AudioDataset objects,
    or as paths (that are loaded with AudioDataset, with the given dataset arguments)."""
    _fields_to_compare_ = ['_output_pattern']

    def __init__(self,
                 datasets: List[AudioDataset | str | Path] | None = None,
                 weights: List[float] | None = None,
                 max_samples: int | None = None,
                 use_cache: bool = False,
                 transforms: TransformType = None,
                 _dataset_preset: tuple | None = None,
                 path: List[str | Path] | None = None,
                 features_after_transforms: bool | None = None,
                 **dataset_kwargs: Any):
        """
        Args:
            datasets (List[AudioDataset | str | Path]): datasets to combine, or their paths.
            weights (List[float] | None, optional): sampling probability of each dataset (normalized) ; if None, items are
                sampled uniformly (datasets are sampled proportionally to their lengths). Defaults to None.
            max_samples (int | None, optional): number of samples per epoch of the sampler. Defaults to None.
            transforms (optional): transforms applied to items, instead of the transforms of each dataset. Defaults to None.
            features_after_transforms (bool | None, optional): compute features on audio after (True) or before (False) its
                transforms (see AudioDataset.features_after_transforms), for every dataset ; None keeps the setting of 
                datasets (True by default). Defaults to None.
            **dataset_kwargs: arguments of datasets given as paths.
        """
        # path is an alias of datasets (see AudioDataset)
        datasets = datasets if datasets is not None else path
        if datasets is None:
            raise ValueError("CombinedAudioDataset requires datasets")
        datasets = [self._load_dataset(d, _dataset_preset, dataset_kwargs) for d in checklist(datasets)]
        if features_after_transforms is not None:
            # datasets are copied if needed, such that given dataset objects are not modified
            datasets = [d.configure(features_after_transforms=features_after_transforms) for d in datasets]
        elif len(set(d.features_after_transforms for d in datasets)) > 1:
            logging.warning("combined datasets have different features_after_transforms settings ; batch features follow the first one")
        if len(datasets) < 2:
            raise ValueError("CombinedAudioDataset must have at least 2 datasets as inputs")
        self._compare_datasets(datasets)
        if weights is not None:
            assert len(weights) == len(datasets), "datasets and weights must be the same size"
            assert all(w >= 0 for w in weights) and sum(weights) > 0, "weights must be positive"
        self._init_common(transforms=transforms, output_pattern=datasets[0]._output_pattern, output_type=datasets[0]._output_type,
                          channels=datasets[0]._n_channels, max_samples=max_samples, 
                          features_after_transforms=datasets[0].features_after_transforms)
        self._weights = weights
        self._datasets = datasets
        self._dataset_map = cumsum([len(d) for d in self._datasets])
        if use_cache: logging.warning("use_cache is not implemented yet in CombinedAudioDataset")
        torch.utils.data.Dataset.__init__(self)

    @staticmethod
    def _load_dataset(dataset, preset, dataset_kwargs) -> AudioDataset:
        if isinstance(dataset, AudioDataset):
            if dataset_kwargs: logging.warning("dataset arguments %s are ignored for given dataset objects"%list(dataset_kwargs.keys()))
            return dataset
        preset, preset_overrides = preset or (None, None)
        return AudioDataset(dataset, preset=preset, preset_overrides=preset_overrides, **dataset_kwargs)

    @classmethod
    def _init_from_preset(cls, preset, preset_overrides, *args, **kwargs):
        # presets are given to the combined datasets
        return type.__call__(cls, *args, _dataset_preset=(preset, preset_overrides), **kwargs)

    def _compare_datasets(self, datasets):
        for field in self._fields_to_compare_:
            values = [getattr(d, field) for d in datasets]
            assert len(set(values)) == 1, "combined datasets must have the same %s, got %s"%(field.strip("_"), values)
        channels = [d._n_channels for d in datasets]
        if len(set(channels)) > 1:
            logging.warning("combined datasets have different numbers of channels (%s) : items may not be batched together"%channels)

    def _reset_map(self):
        self._dataset_map = cumsum([len(d) for d in self._datasets])

    # items

    def __len__(self):
        return self._dataset_map[-1] if self._dataset_map else 0

    def _locate(self, index: int) -> tuple:
        """returns (dataset index, item index in dataset) of an item."""
        if hasattr(index, "item"): index = index.item()
        if index < 0: index += len(self)
        if not (0 <= index < len(self)):
            raise IndexError("index %d out of range for dataset of length %d"%(index, len(self)))
        target_dataset = bisect.bisect_right(self._dataset_map, index)
        if target_dataset > 0: index -= self._dataset_map[target_dataset - 1]
        return target_dataset, index

    def _fragment_from_index(self, index):
        target_dataset, index = self._locate(index)
        return self._datasets[target_dataset]._fragment_from_index(index)

    def get(self, index, output_pattern=None, transforms=None):
        """returns the item at index, with the output pattern and transforms of its dataset (unless given, or
        unless the combined dataset has transforms). Slices and sequences of indices return batches."""
        if is_multi_index(index):
            return collate_items([self.get(i, output_pattern=output_pattern, transforms=transforms) for i in multi_index_to_list(index, len(self))])
        global_index = int(index.item() if hasattr(index, "item") else index) % max(1, len(self))
        target_dataset, index = self._locate(index)
        dataset = self._datasets[target_dataset]
        if transforms is None:
            transforms = self._transforms if self._transforms is not None else dataset.transforms
        return dataset.get(index, output_pattern=output_pattern or self._output_pattern, transforms=transforms, _global_index=global_index)

    @AudioDataset.output_pattern.setter
    def output_pattern(self, pattern):
        assert isinstance(pattern, str)
        self._output_pattern = pattern
        for d in self._datasets: d.output_pattern = pattern

    @AudioDataset.features_after_transforms.setter
    def features_after_transforms(self, value: bool):
        self._features_after_transforms = bool(value)
        for d in self._datasets: d.features_after_transforms = value

    def configure(self, **kwargs) -> "CombinedAudioDataset":
        """returns a combined dataset of configured datasets (e.g. chunk_length of online datasets). Loading arguments
        that do not apply to a dataset (e.g. batch_by for pre-processed datasets) are ignored for this dataset."""
        if len(kwargs) == 0:
            return self
        dataset = copy.copy(self)
        if kwargs.get('features_after_transforms') is not None:
            dataset._features_after_transforms = bool(kwargs['features_after_transforms'])
        dataset._datasets = [self._configure_dataset(d, kwargs) for d in self._datasets]
        dataset._reset_map()
        return dataset

    @staticmethod
    def _configure_dataset(dataset, kwargs):
        import inspect
        params = inspect.signature(dataset.configure).parameters
        accepted = {k: v for k, v in kwargs.items() if k in params}
        ignored = set(kwargs) - set(accepted)
        if ignored: logging.debug("loading arguments %s ignored for %s"%(ignored, type(dataset).__name__))
        return dataset.configure(**accepted) if accepted else dataset

    def _check_loader_arguments(self, batch_size, loader_kwargs):
        for d in self._datasets:
            d._check_loader_arguments(batch_size, loader_kwargs)

    def item_labels(self, features, n_bins: int = 10):
        """labels of items of each dataset, as (dataset index, label) ; None for items without label."""
        labels = []
        for i, d in enumerate(self._datasets):
            labels.extend([None if l is None else (i, l) for l in d.item_labels(features, n_bins=n_bins)])
        return labels

    def sampling_weights(self, balance_by=None, balance_power: float = 1., n_bins: int = 10, missing: str = "exclude") -> torch.Tensor:
        """sampling weights of items : labels of balance_by are balanced within each dataset, and datasets are sampled with 
        their weights (or proportionally to their lengths if weights is None)."""
        weights = []
        total = sum(len(d) for d in self._datasets)
        dataset_weights = self._weights if self._weights is not None else [len(d) for d in self._datasets]
        norm = sum(dataset_weights)
        for d, w in zip(self._datasets, dataset_weights):
            if balance_by is not None:
                item_weights = d.sampling_weights(balance_by, balance_power=balance_power, n_bins=n_bins, missing=missing)
            else:
                item_weights = torch.full((len(d),), 1. / max(1, len(d)), dtype=torch.float64)
            weights.append(item_weights * (w / norm))
        return torch.cat(weights)

    def get_sampler(self, replacement: bool | None = None, max_samples: int | None = None, valid: bool = False, balance_by=None, 
                    balance_power: float = 1., n_bins: int = 10, missing: str = "exclude", curriculum=None, weights=None, 
                    seed: int | None = None, rank: int | None = None, world_size: int | None = None):
        """weighted sampler : items are sampled uniformly if weights is None, and datasets are sampled with their 
        weights otherwise. Labels of balance_by are balanced within each dataset ; see AudioDataset.get_sampler."""
        if all(x is None for x in [balance_by, curriculum, weights, seed, rank, world_size]):
            max_samples = max_samples or self._max_samples or len(self)
            item_weights = self.sampling_weights().tolist()
            generator = None if valid else torch.Generator().manual_seed(42)
            return torch.utils.data.WeightedRandomSampler(item_weights, max_samples, replacement=bool(replacement), generator=generator)
        if weights is None:
            weights = self.sampling_weights(balance_by, balance_power=balance_power, n_bins=n_bins, missing=missing)
        return super().get_sampler(replacement=replacement, max_samples=max_samples, curriculum=curriculum, weights=weights, 
                                   seed=seed, rank=rank, world_size=world_size)

    # properties

    @property
    def datasets(self) -> List[AudioDataset]:
        return list(self._datasets)

    @property
    def weights(self) -> List[float] | None:
        return None if self._weights is None else list(self._weights)

    @property
    def path(self) -> List[str]:
        return [d.path for d in self._datasets]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {'n_items': len(self), 'weights': self.weights, 'chunk_length': self.chunk_length,
                'datasets': [d.metadata for d in self._datasets]}

    @property
    def features(self) -> Dict[str, Any]:
        """features of the combined datasets, by name."""
        features = {}
        for d in self._datasets:
            for f in d._feature_objects():
                features.setdefault(getattr(f, "feature_name", type(f).__name__.lower()), f)
        return features

    @property
    def feature_hash(self) -> List[Dict]:
        return [d.feature_hash for d in self._datasets]

    @property
    def chunk_length(self) -> int | None:
        """chunk length of items if it is the same for every dataset, None otherwise."""
        lengths = {d.chunk_length for d in self._datasets}
        return lengths.pop() if len(lengths) == 1 else None

    @property
    def keys(self) -> List[List[str]]:
        return [d.keys for d in self._datasets]

    def to(self, device):
        for d in self._datasets: d.to(device)
        return self

    def _computed_features(self):
        computed = {}
        for d in self._datasets:
            for name, f in d._computed_features().items():
                computed.setdefault(name, f)
        return computed

    def _without_features(self, names):
        dataset = copy.copy(self)
        dataset._datasets = [d._without_features(names) for d in self._datasets]
        return dataset

    def _set_worker_outputs(self, output_pattern, transforms):
        self._output_pattern = output_pattern
        self._transforms = transforms
        for d in self._datasets:
            d._set_worker_outputs(output_pattern, transforms)

    @property
    def feature_devices(self) -> set:
        return set().union(*[d.feature_devices for d in self._datasets])

    # files

    @property
    def files(self) -> List[Dict[str, Any]]:
        """files of the combined datasets, in order, with the index of their dataset (dataset key)."""
        files = []
        for i, d in enumerate(self._datasets):
            for f in d.files:
                files.append({**f, 'dataset': i})
        return files

    def _locate_file(self, file_idx: int) -> tuple:
        counts = cumsum([len(d.files) for d in self._datasets])
        if file_idx < 0: file_idx += counts[-1]
        if not (0 <= file_idx < counts[-1]):
            raise IndexError("file index %d out of range (%d files)"%(file_idx, counts[-1]))
        target = bisect.bisect_right(counts, file_idx)
        return target, file_idx - (counts[target - 1] if target > 0 else 0)

    def load_audio(self, file_idx: int, start: float = 0., length: int | float | None = None, pad_mode=None) -> torch.Tensor:
        target, file_idx = self._locate_file(file_idx)
        return self._datasets[target].load_audio(file_idx, start=start, length=length, pad_mode=pad_mode)

    def get_fragment(self, file_idx: int, start: float = 0., length: int | float | None = None, features=None, **kwargs):
        target, file_idx = self._locate_file(file_idx)
        return self._datasets[target].get_fragment(file_idx, start=start, length=length, features=features, **kwargs)

    # partitions

    def _from_partitions(self, datasets: List[AudioDataset]) -> "CombinedAudioDataset":
        """returns a combined dataset of partitions of the combined datasets."""
        assert len(datasets) == len(self._datasets)
        subdataset = copy.copy(self)
        subdataset._datasets = list(datasets)
        subdataset._reset_map()
        subdataset.parent = self
        if self._max_samples:
            subdataset._max_samples = int(self._max_samples * (len(subdataset) / len(self)))
        return subdataset

    def _check_feature_in_subdatasets(self, features):
        for f in checklist(features):
            for i, d in enumerate(self._datasets):
                assert f in d.feature_hash or f in [getattr(o, "feature_name", None) for o in d._feature_objects()] or f in getattr(d, "features", {}), \
                    "feature %s not present in dataset %d (path : %s)"%(f, i, d.path)

    def split(self, partitions, features=None, **kwargs):
        """splits each combined dataset with the same partitions (see AudioDataset.split), and returns combined datasets
        of the corresponding partitions."""
        assert self.parent is None, "dataset is already a partition"
        if features is not None:
            self._check_feature_in_subdatasets(features)
        partitions_list = [d.split(partitions, features=features, **kwargs) for d in self._datasets]
        return {p: self._from_partitions([d[p] for d in partitions_list]) for p in partitions_list[0].keys()}

    def load_partition(self, name: str | None = None, check: bool = False):
        partitions_list = [d.load_partition(name, check=check) for d in self._datasets]
        return {p: self._from_partitions([d[p] for d in partitions_list]) for p in partitions_list[0].keys()}


__all__ = ['CombinedAudioDataset']
