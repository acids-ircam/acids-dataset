import inspect
import math
import random
import numpy as np
import difflib
from absl import logging
import torch
from abc import abstractmethod, ABCMeta
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, TypeVar, TYPE_CHECKING
from .. import transforms
from .utils import _outs_from_pattern, _transform_outputs, is_multi_index, multi_index_to_list, collate_items

TransformType = Optional[transforms.Transform | List[transforms.Transform] | Dict[str, transforms.Transform]]

if TYPE_CHECKING:
    # AudioDataset(...) is valid : _AudioDatasetMeta dispatches it to a concrete subclass. Static checkers cannot
    # see this, and would report AudioDataset as abstract at each instantiation : abstract methods are hence only
    # declared as such at runtime (ABCMeta still prevents incomplete subclasses from being instantiated).
    _F = TypeVar("_F", bound=Callable[..., Any])
    def _abstractmethod(func: _F) -> _F:
        return func
else:
    _abstractmethod = abstractmethod

_DATALOADER_ARGS = set(inspect.signature(torch.utils.data.DataLoader.__init__).parameters.keys()) - {'self', 'dataset'}


def is_preprocessed_path(path: str | Path) -> bool:
    """returns True if path is a pre-processed dataset folder (i.e. written by a dataset writer)."""
    if isinstance(path, (list, tuple)):
        return False
    path = Path(path)
    return path.is_dir() and (path / "metadata.yaml").exists() and (path / "data.mdb").exists()


def _channels_of_preprocessed(path: Path) -> set:
    """channels of a pre-processed dataset : from its metadata, or (if channels were kept, i.e. None) from the waveform
    buffer of a chunk of each original file (only buffer shapes are read)."""
    from .. import get_metadata_from_path, get_loader_from_path
    metadata = get_metadata_from_path(path)
    if metadata.get('channels') is not None:
        return {int(metadata['channels'])}
    if not metadata.get('waveform', True):
        raise ValueError("dataset %s was pre-processed without waveform, and without channels in its metadata"%path)
    loader = get_loader_from_path(path)
    channels = set()
    for keys in loader.feature_hash['original_path'].values():
        if len(keys) == 0: continue
        key = keys[0]
        key = loader.keygen.from_str(key) if isinstance(key, str) else key
        buffer = loader[key].get_buffer('waveform')
        channels.add(int(buffer.shape[0]) if len(buffer.shape) > 1 else 1)
    return channels


def _audio_files(dir_path: Path, valid_exts=None, filters=(), exclude=()) -> List[Path]:
    """audio files of a folder (recursive), in a single walk ; filters and exclude are glob-like patterns of paths 
    relative to the folder (as writers.utils.audio_paths_from_dir)."""
    import os, fnmatch
    from ..writers.utils import _VALID_EXTS
    exts = {e.lower() for e in (valid_exts or _VALID_EXTS)}
    files = []
    for root, _, names in os.walk(dir_path, followlinks=True):
        for name in names:
            if os.path.splitext(name)[1].lower() not in exts: continue
            rel = os.path.relpath(os.path.join(root, name), dir_path)
            if filters and not any(fnmatch.fnmatch(rel, f) for f in filters): continue
            if any(fnmatch.fnmatch(rel, e) for e in exclude): continue
            files.append(Path(root) / name)
    return files


CHANNELS_FILE_NAME = ".ad_channels"
CHANNELS_FILE_VERSION = 1


def channels_file_path(dataset_path: str | Path, index_dir: str | Path | None = None) -> Path:
    """location of the channels file of a raw audio folder (see get_channels_from_path) : ".ad_channels" in the folder, 
    or "<folder_name>.ad_channels" in index_dir."""
    dataset_path = Path(dataset_path).resolve()
    if index_dir is None:
        return dataset_path / CHANNELS_FILE_NAME
    return Path(index_dir).resolve() / f"{dataset_path.name}{CHANNELS_FILE_NAME}"


def get_channels_from_path(path: str | Path | List[str | Path], index_dir: str | Path | None = None, valid_exts: List[str] | None = None,
                           filters: List[str] = [], exclude: List[str] = [], n_threads: int = 16, write: bool = False,
                           force: bool = False) -> set:
    """returns the set of channel counts of the audio of a dataset location, without decoding audio :
        - pre-processed dataset folders : channels of their metadata (or, if channels were kept at pre-processing, the 
            shapes of a stored chunk of each original file) ;
        - raw audio folders : channels of audio files, read from the index of the folder (see OnlineAudioDataset) for 
            unmodified files, and from file headers otherwise (in n_threads threads). Index files are not written ;
        - audio files : channels of their header ;
        - lists of locations : union of their channels.

    For raw audio folders, the result can be written in a channels file (write=True) : a hidden ".ad_channels" file in 
    the folder, or "<folder_name>.ad_channels" in index_dir if given. If this file exists (and was written with the same
    valid_exts, filters and exclude), channels are read from it without scanning the folder, unless force=True. The 
    channels file is not updated when files are added or modified : give force=True to scan the folder again.

    Args:
        path (str | Path | List[str | Path]): dataset location(s).
        index_dir (str | Path | None, optional): index folder of raw folders (see OnlineAudioDataset). Defaults to None.
        valid_exts (List[str] | None, optional): audio extensions of raw folders ; None for default ones. Defaults to None.
        filters (List[str], optional): glob-like patterns of audio paths of raw folders. Defaults to [].
        exclude (List[str], optional): glob-like patterns of excluded audio paths of raw folders. Defaults to [].
        n_threads (int, optional): threads reading file headers. Defaults to 16.
        write (bool, optional): write the channels of raw audio folders in their channels file. Defaults to False.
        force (bool, optional): scan raw audio folders even if a channels file exists (and overwrite it with write). 
            Defaults to False.
    """
    if isinstance(path, (list, tuple)):
        return set().union(*[get_channels_from_path(p, index_dir=index_dir, valid_exts=valid_exts, filters=filters, 
                                                    exclude=exclude, n_threads=n_threads, write=write, force=force) for p in path])
    from ..utils import audio_info
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("%s does not exist"%path)
    if is_preprocessed_path(path):
        return _channels_of_preprocessed(path)
    if path.is_file():
        return {audio_info(path)['channels']}
    import json
    channels_path = channels_file_path(path, index_dir)
    scan_options = {'valid_exts': sorted(valid_exts) if valid_exts else None, 'filters': _checklist(filters), 'exclude': _checklist(exclude)}
    if not force and channels_path.exists():
        try:
            with open(channels_path, "r") as f:
                stored = json.load(f)
            if stored.get('version') == CHANNELS_FILE_VERSION and stored.get('options') == scan_options:
                return set(int(c) for c in stored['channels'])
        except Exception as e:
            logging.warning("could not read channels file %s (%s) ; scanning %s"%(channels_path, e, path))
    channels = _scan_channels(path, index_dir, valid_exts, filters, exclude, n_threads)
    if write:
        try:
            import os
            channels_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = channels_path.with_name(channels_path.name + ".tmp")
            with open(tmp_path, "w") as f:
                json.dump({'version': CHANNELS_FILE_VERSION, 'dataset_path': str(path.resolve()), 'options': scan_options,
                           'channels': sorted(channels)}, f, indent=1)
            os.replace(tmp_path, channels_path)
        except OSError as e:
            logging.warning("could not write channels file %s (%s)"%(channels_path, e))
    return channels


def _scan_channels(path: Path, index_dir, valid_exts, filters, exclude, n_threads) -> set:
    """channels of the audio files of a raw folder, from its index for unmodified files and from headers otherwise."""
    from ..utils import audio_info
    from .online import OnlineAudioDataset
    index_path = OnlineAudioDataset.index_path_for(path, index_dir)
    cached = {}
    if index_path.exists():
        try:
            import json
            with open(index_path, "r") as f:
                index = json.load(f)
            if index.get('version') == OnlineAudioDataset.index_version:
                cached = index.get('files', {})
        except Exception:
            cached = {}
    channels, to_read = set(), []
    for audio_file in _audio_files(path, valid_exts=valid_exts, filters=_checklist(filters), exclude=_checklist(exclude)):
        entry = cached.get(str(audio_file.relative_to(path)))
        if entry is not None and 'channels' in entry:
            stat = audio_file.stat()
            if entry.get('size') == stat.st_size and entry.get('mtime') == stat.st_mtime:
                channels.add(int(entry['channels']))
                continue
        to_read.append(audio_file)

    def read(audio_file):
        try:
            return audio_info(audio_file)['channels']
        except Exception as e:
            logging.warning("could not read file %s : %s"%(audio_file, e))
            return None
    if len(to_read) > 0:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, min(n_threads, len(to_read)))) as executor:
            channels.update(c for c in executor.map(read, to_read) if c is not None)
    return channels


def _checklist(x) -> list:
    return [x] if isinstance(x, str) else list(x or [])


def chunk_length_to_samples(chunk_length: int | float, sr: int) -> int:
    """chunk lengths follow parsers convention : int are samples, float are seconds."""
    if isinstance(chunk_length, int):
        return chunk_length
    return int(round(chunk_length * sr))


class _AudioDatasetMeta(ABCMeta):
    """chooses the dataset class from the path when AudioDataset is instantiated, and parses presets."""
    def __call__(cls, *args, preset: str | Path | None = None, preset_overrides: List[str] | None = None, **kwargs):
        if cls is AudioDataset:
            path = args[0] if len(args) > 0 else kwargs.get('path', kwargs.get('db_path'))
            cls = AudioDataset.dataset_class_from_path(path)
        if preset is None and not preset_overrides:
            return super().__call__(*args, **kwargs)
        return cls._init_from_preset(preset, preset_overrides or [], *args, **kwargs)


class AudioDataset(torch.utils.data.Dataset, metaclass=_AudioDatasetMeta):
    """Base class for audio datasets. Instantiating AudioDataset directly returns a PreprocessedAudioDataset
    if path is a pre-processed dataset folder, and an OnlineAudioDataset otherwise (raw audio folders). 
    
    Datasets can be instantiated with a preset (see acids_dataset.presets) with the preset keyword, and 
    additional gin bindings with preset_overrides. Explicit arguments override preset values."""

    # dataset arguments that override preset values
    _preset_values = {}

    if TYPE_CHECKING:
        # signature of AudioDataset(...) for static checkers (arguments are those of the dispatched subclass)
        def __init__(self, path: str | Path | List[str | Path] | None = None, *args: Any,
                     preset: str | Path | None = None, preset_overrides: List[str] | None = None, **kwargs: Any) -> None: ...

    @classmethod
    def _check_kwargs(cls, kwargs: Dict[str, Any], other_class: type) -> None:
        """raises an error for unknown arguments (with suggestions), and warns for arguments of the other dataset class 
        (such that the same arguments can be given for raw and pre-processed folders)."""
        def params(c):
            return [p for p in inspect.signature(c.__init__).parameters if p not in ["self", "kwargs"]]
        own_params, other_params = params(cls), params(other_class)
        unknown = []
        for k in kwargs:
            if k in other_params:
                logging.warning("argument %s is ignored for %s"%(k, cls.__name__))
            else:
                unknown.append(k)
        if len(unknown) > 0:
            suggestions = {k: difflib.get_close_matches(k, own_params, n=3, cutoff=0.5) for k in unknown}
            message = ", ".join(k + (" (did you mean %s?)"%" or ".join(s) if s else "") for k, s in suggestions.items())
            raise TypeError("%s got unexpected arguments : %s"%(cls.__name__, message))

    @classmethod
    def _preset_values_from_kwargs(cls, kwargs) -> Dict[str, Any]:
        return {value: kwargs[arg] for arg, value in cls._preset_values.items() if arg in kwargs}

    @classmethod
    def _init_from_preset(cls, preset, preset_overrides, *args, **kwargs):
        from ..presets import preset_env
        with preset_env(preset, overrides=preset_overrides, values=cls._preset_values_from_kwargs(kwargs)):
            return type.__call__(cls, *args, **kwargs)

    @staticmethod
    def dataset_class_from_path(path: str | Path | List[str | Path]) -> type:
        from .simple import PreprocessedAudioDataset
        from .online import OnlineAudioDataset
        if path is None:
            raise ValueError("AudioDataset requires a path.")
        if isinstance(path, (list, tuple)):
            if any(is_preprocessed_path(p) for p in path):
                # pre-processed datasets cannot be merged : they are combined
                from .combined import CombinedAudioDataset
                return CombinedAudioDataset
            return OnlineAudioDataset
        if is_preprocessed_path(path):
            return PreprocessedAudioDataset
        return OnlineAudioDataset

    def _init_common(self,
                     transforms: TransformType = None,
                     output_pattern: str = "waveform",
                     output_type: str = "torch",
                     channels: int | None = 1,
                     subindices = None,
                     max_samples: int | None = None,
                     features_after_transforms: bool = True) -> None:
        self._transforms = transforms
        self._features_after_transforms = bool(features_after_transforms)
        self._output_pattern = output_pattern
        self._output_type = output_type
        self._n_channels = channels
        self._subindices = subindices
        self._max_samples = max_samples
        self._parent = None
        self._partitions = {}

    # abstract methods

    @_abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @_abstractmethod
    def _fragment_from_index(self, index):
        """returns the AudioFragment corresponding to index."""
        raise NotImplementedError()

    @property
    @_abstractmethod
    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @property
    @_abstractmethod
    def features(self):
        raise NotImplementedError()

    @property
    @_abstractmethod
    def chunk_length(self) -> int | None:
        """chunk length (in samples) of dataset items."""
        raise NotImplementedError()

    @property
    @_abstractmethod
    def files(self) -> List[Dict[str, Any]]:
        """audio files of the dataset, as dicts with (at least) audio_path, duration, sr and channels keys. File indices 
        of load_audio and get_fragment refer to this list."""
        raise NotImplementedError()

    @_abstractmethod
    def load_audio(self, file_idx: int, start: float = 0., length: int | float | None = None, pad_mode = None) -> torch.Tensor:
        """loads audio of a given file at the sampling rate of the dataset, as a (channels, samples) tensor.

        Args:
            file_idx (int): file index (in self.files).
            start (float, optional): start position (seconds). Defaults to 0.
            length (int | float | None, optional): length, in samples (int) or seconds (float). None for the whole file. Defaults to None.
            pad_mode (PadMode | str | None, optional): padding mode if the requested audio exceeds the file. Defaults to None.
        """
        raise NotImplementedError()

    @_abstractmethod
    def get_fragment(self, file_idx: int, start: float = 0., length: int | float | None = None, features = None, **kwargs):
        """loads audio of a given file (see load_audio) as a fragment, with features computed on the loaded audio 
        (None for dataset features)."""
        raise NotImplementedError()

    @_abstractmethod
    def split(self, partitions, **kwargs):
        raise NotImplementedError()

    @_abstractmethod
    def load_partition(self, name: str | None, check: bool = False):
        raise NotImplementedError()

    def configure(self, features_after_transforms: bool | None = None, **kwargs) -> "AudioDataset":
        """returns a dataset configured with given loading parameters (e.g. chunk_length, features_after_transforms)."""
        if len(kwargs) > 0:
            raise TypeError("%s got unexpected loading arguments : %s"%(type(self).__name__, list(kwargs.keys())))
        if features_after_transforms is None or bool(features_after_transforms) == self.features_after_transforms:
            return self
        import copy
        dataset = copy.copy(self)
        dataset.features_after_transforms = features_after_transforms
        return dataset

    @property
    def features_after_transforms(self) -> bool:
        """if True (default), features computed by the dataset (and batch features) describe the augmented audio, i.e. the
        waveform output after its transforms ; if False, they describe the loaded audio, before transforms. Pre-processed
        features are read from the database, and always describe the original audio."""
        return getattr(self, "_features_after_transforms", True)

    @features_after_transforms.setter
    def features_after_transforms(self, value: bool) -> None:
        self._features_after_transforms = bool(value)

    @staticmethod
    def _assign_by_ratio(groups: List[Any], weights: List[float], ratios: Dict[str, float]) -> Dict[str, List[Any]]:
        """assigns ordered groups to partitions such that partition weights follow ratios ; each group is 
        assigned to the partition containing its cumulative weight midpoint."""
        total = sum(weights)
        assignment = {k: [] for k in ratios}
        bounds, current = [], 0.
        for k, r in ratios.items():
            current += r
            bounds.append((current, k))
        cum = 0.
        for g, w in zip(groups, weights):
            midpoint = (cum + w / 2) / total if total > 0 else 0.
            cum += w
            for bound, k in bounds:
                if midpoint < bound:
                    assignment[k].append(g)
                    break
        return assignment

    @staticmethod
    def _assign_stratified(groups: List[Any], weights: List[float], values: List[float], ratios: Dict[str, float], 
                           n_bins: int = 10, rng=random) -> Dict[str, List[Any]]:
        """assigns groups to partitions such that the distribution of a continuous value is similar in every partition. 
        Groups are sorted in n_bins quantile bins of their value (histogram), shuffled within each bin, and assigned in 
        order of value to the partition with the largest weight deficit. If ratios sum below 1, the remaining ratio is 
        an implicit partition whose groups are left out."""
        values = np.asarray(values, dtype=np.float64)
        n_bins = max(1, min(int(n_bins), len(groups)))
        edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1))) if len(groups) else np.array([])
        bins = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, max(0, len(edges) - 2)) if len(edges) > 1 else np.zeros(len(groups), dtype=int)
        order = []
        for b in sorted(set(bins.tolist())):
            members = [i for i in range(len(groups)) if bins[i] == b]
            rng.shuffle(members)
            order.extend(members)
        targets = dict(ratios)
        leftover = 1. - sum(ratios.values())
        if leftover > 1e-6:
            targets[None] = leftover
        current = {k: 0. for k in targets}
        assignment = {k: [] for k in targets}
        cumulative = 0.
        for i in order:
            cumulative += weights[i]
            k = max(targets, key=lambda k: targets[k] * cumulative - current[k])
            assignment[k].append(groups[i])
            current[k] += weights[i]
        assignment.pop(None, None)
        return assignment

    @staticmethod
    def _reduce_feature_value(value) -> float:
        """reduces a feature value of an item to a scalar (mean over finite values), for stratified partitions."""
        if torch.is_tensor(value): value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        value = value[np.isfinite(value)]
        return float(value.mean()) if value.size else float("nan")

    @staticmethod
    def _assign_balanced(groups: List[Any], weights: List[float], ratios: Dict[str, float], tolerance: float = 0.1) -> Dict[str, List[Any]]:
        """assigns groups greedily (heaviest first) to the partition with the largest relative deficit. If ratios 
        sum below 1, the remaining ratio is an implicit partition whose groups are left out."""
        total = sum(weights)
        targets = {k: r * total for k, r in ratios.items()}
        leftover = 1. - sum(ratios.values())
        if leftover > 1e-6:
            targets[None] = leftover * total
        current = {k: 0. for k in targets}
        assignment = {k: [] for k in targets}
        for w, g in sorted(zip(weights, groups), key=lambda x: -x[0]):
            k = max([k for k in targets if targets[k] > 0], key=lambda k: (targets[k] - current[k]) / targets[k])
            assignment[k].append(g)
            current[k] += w
        for k in ratios:
            if targets[k] > 0 and abs(current[k] - targets[k]) > tolerance * targets[k]:
                logging.warning("partition %s : weight %.2f for target %.2f (tolerance %.2f)"%(k, current[k], targets[k], tolerance))
        assignment.pop(None, None)
        return assignment

    # common methods

    @staticmethod
    def _build_fragment(fragment_class, wav, sr, audio_path, start, file_length, features, output_type="torch", bformat="float32", metadata=None):
        """builds a fragment from loaded audio, and computes features on it."""
        fragment = fragment_class(
            audio_path = audio_path,
            audio = wav,
            start_pos = start,
            length = wav.shape[-1] / sr,
            file_length = file_length,
            bformat = bformat,
            sr = sr,
            # features are computed on numpy outputs, as in LMDBWriter
            output_type = "numpy",
            **{k: v for k, v in (metadata or {}).items() if v is not None}
        )
        for f in features:
            f.extract(fragment=fragment)
        fragment.output_type = output_type
        return fragment

    def __getitem__(self, index):
        return self.get(index)

    def get(self, index, output_pattern=None, transforms=None, _global_index=None):
        """returns the item at index. Slices and sequences of indices return a batch of items (collated as in 
        torch.utils.data.DataLoader, or as a list if items cannot be collated)."""
        if is_multi_index(index):
            indices = multi_index_to_list(index, len(self))
            return collate_items([self.get(i, output_pattern=output_pattern, transforms=transforms) for i in indices])
        if hasattr(index, "item") and not isinstance(index, (str, bytes)):
            # 0-d tensors / numpy integers
            index = index.item()
        output_pattern = output_pattern or self.output_pattern
        transforms = transforms or self.transforms
        fg = self._fragment_from_index(index)
        # "index" entries of output patterns give the index of the item (in combined datasets if any)
        item_index = _global_index if _global_index is not None else (index % len(self) if isinstance(index, int) else None)
        outs = _outs_from_pattern(fg, output_pattern, index=item_index)
        outs = _transform_outputs(outs, transforms, self._n_channels)
        return outs

    def item_labels(self, features, n_bins: int = 10) -> List[Any]:
        """label of each item for given features : values of categorical (hashed) features, and histogram (quantile) 
        bins of continuous features, as tuples for several features ; None for items without value."""
        raise NotImplementedError("%s does not give item labels"%type(self).__name__)

    @staticmethod
    def _combine_labels(per_feature: List[List[Any]]) -> List[Any]:
        if len(per_feature) == 1:
            return per_feature[0]
        return [None if any(v is None for v in values) else tuple(values) for values in zip(*per_feature)]

    def sampling_weights(self, balance_by, balance_power: float = 1., n_bins: int = 10, missing: str = "exclude") -> torch.Tensor:
        """sampling weights of items, balancing the labels of given features (see acids_dataset.sampling.balance_weights)."""
        from ..sampling import balance_weights
        return balance_weights(self.item_labels(balance_by, n_bins=n_bins), power=balance_power, missing=missing)

    def get_sampler(self, replacement: bool | None = None, max_samples: int | None = None, balance_by=None, balance_power: float = 1., 
                    n_bins: int = 10, missing: str = "exclude", curriculum = None, weights = None, seed: int | None = None, 
                    rank: int | None = None, world_size: int | None = None):
        """returns a sampler of dataset items. Without balancing / curriculum / seed / distribution, items are sampled 
        uniformly (torch RandomSampler). Otherwise, returns an acids_dataset.sampling.WeightedItemSampler, that balances 
        labels of balance_by features (balance_power), follows a curriculum (acids_dataset.sampling.Curriculum), is 
        reproducible with seed, and distributed among processes (rank, world_size ; inferred from torch.distributed if 
        initialized). max_samples is the number of samples per epoch (over all processes)."""
        max_samples = max_samples or self._max_samples
        if all(x is None for x in [balance_by, curriculum, weights, seed, rank, world_size]):
            return torch.utils.data.RandomSampler(self, replacement=bool(replacement), num_samples=max_samples or len(self))
        from ..sampling import WeightedItemSampler, _distributed_info
        if balance_by is not None:
            weights = self.sampling_weights(balance_by, balance_power=balance_power, n_bins=n_bins, missing=missing)
        rank, world_size = _distributed_info(rank, world_size)
        num_samples = None if max_samples is None else int(math.ceil(max_samples / world_size))
        return WeightedItemSampler(len(self), weights=weights, curriculum=curriculum, num_samples=num_samples, 
                                   replacement=True if replacement is None else replacement, seed=seed, rank=rank, world_size=world_size)

    def get_batch_sampler(self, balance_by, classes_per_batch: int, items_per_class: int, num_batches: int | None = None, 
                          n_bins: int = 10, seed: int | None = None, rank: int | None = None, world_size: int | None = None):
        """returns a sampler of batches of classes_per_batch labels of balance_by features, with items_per_class items each 
        (see acids_dataset.sampling.BalancedBatchSampler)."""
        from ..sampling import BalancedBatchSampler
        return BalancedBatchSampler(self.item_labels(balance_by, n_bins=n_bins), classes_per_batch, items_per_class, 
                                    num_batches=num_batches, seed=seed, rank=rank, world_size=world_size)

    def get_loader(self, batch_size: int = 1, shuffle: bool = True, replacement: bool | None = None, max_samples: int | None = None, 
                   batch_features = None, device = None, batch_transforms: bool = False, balance_by = None, balance_power: float = 1., 
                   curriculum = None, seed: int | None = None, balanced_batches: tuple | None = None, sampler_kwargs: Dict[str, Any] | None = None, 
                   **kwargs):
        """returns a DataLoader for the dataset. Keyword arguments of torch.utils.data.DataLoader (num_workers,
        drop_last, ...) are given to the DataLoader ; other ones (chunk_length, features_after_transforms, ...) are given to self.configure. 
        If features run on an accelerator (see to), workers are started with spawn, as CUDA / MPS cannot be used in 
        forked processes.

        Args:
            batch_features (List[str | AcidsDatasetFeature] | str | None, optional): features computed on whole batches in the 
                main process (on device), instead of item by item in workers : names of features computed by the dataset, 
                feature objects (e.g. to compute new features on pre-processed datasets), or "auto" for features on 
                accelerators. Batch features are computed on the waveform output after its transforms (or before them,
                see features_after_transforms). Defaults to None.
            device (str | torch.device | None, optional): device of batches, batch features and batch transforms. Defaults to None.
            batch_transforms (bool, optional): apply transforms on batches (on device) in the main process, rather than on 
                items in workers. Defaults to False.
            balance_by (str | List[str] | None, optional): features whose labels are balanced (see get_sampler). Defaults to None.
            balance_power (float, optional): balancing power (1 : labels are sampled uniformly, 0 : no balancing). Defaults to 1.
            curriculum (Curriculum | None, optional): curriculum of sampled items (see acids_dataset.sampling). Defaults to None.
            seed (int | None, optional): seed of the sampler (reproducible, and identical across processes). Defaults to None.
            balanced_batches (tuple | None, optional): (classes_per_batch, items_per_class) : batches of classes_per_batch labels of 
                balance_by, with items_per_class items each (batch_size is then ignored). Defaults to None.
            sampler_kwargs (dict | None, optional): additional arguments of get_sampler (n_bins, missing, rank, world_size...).
        Returns a torch.utils.data.DataLoader, or a BatchFeatureLoader (that behaves as a DataLoader) if batch_features 
        or batch_transforms are given.
        """
        loader_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in _DATALOADER_ARGS}
        dataset = self.configure(**kwargs)
        sampler_kwargs = dict(sampler_kwargs or {})
        # samplers are built on the configured dataset (that computes every feature, e.g. for continuous balancing)
        batch_sampler, sampler = None, loader_kwargs.pop('sampler', None)
        if balanced_batches is not None:
            assert balance_by is not None, "balanced_batches require balance_by"
            classes_per_batch, items_per_class = balanced_batches
            batch_sampler = dataset.get_batch_sampler(balance_by, classes_per_batch, items_per_class, seed=seed, **sampler_kwargs)
        elif sampler is None and shuffle:
            sampler = dataset.get_sampler(replacement=replacement, max_samples=max_samples, balance_by=balance_by, balance_power=balance_power, 
                                          curriculum=curriculum, seed=seed, **sampler_kwargs)
        processor = None
        if batch_features or batch_transforms:
            from .batch import BatchFeatureProcessor
            processor = BatchFeatureProcessor(dataset, features=batch_features, device=device, batch_transforms=batch_transforms)
            dataset = processor.worker_dataset
        dataset._check_loader_arguments(batch_size, loader_kwargs)
        if loader_kwargs.get('num_workers', 0) > 0 and loader_kwargs.get('multiprocessing_context') is None and dataset.feature_devices - {"cpu"}:
            logging.info("features on %s : data loader workers are started with spawn"%(dataset.feature_devices - {"cpu"}))
            loader_kwargs['multiprocessing_context'] = "spawn"
        if batch_sampler is not None:
            for k in ['drop_last']: loader_kwargs.pop(k, None)
            loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, **loader_kwargs)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, **loader_kwargs)
        if processor is not None:
            from .batch import BatchFeatureLoader
            return BatchFeatureLoader(loader, processor)
        return loader

    # feature sampling

    @staticmethod
    def _parse_size(size: int | float | str | None) -> int | None:
        """parses a size in bytes, given as a number or as a string with a binary unit (e.g. "1G", "500M", "1.5GB", "64k")."""
        if size is None or isinstance(size, (int, float)):
            return None if size is None else int(size)
        import re
        result = re.match(r"^\s*([\d\.]+)\s*([kKmMgGtT]?)i?[bB]?\s*$", str(size))
        if result is None:
            raise ValueError("could not parse size %s (expected e.g. 1G, 500M, 64k, or a number of bytes)"%size)
        value, unit = result.groups()
        return int(float(value) * 1024 ** "_KMGT".index(unit.upper() or "_"))

    @staticmethod
    def _nbytes(obj) -> int:
        """size in memory of the tensors / arrays of a (possibly nested) batch."""
        if torch.is_tensor(obj):
            return obj.element_size() * obj.nelement()
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        if isinstance(obj, dict):
            return sum(AudioDataset._nbytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(AudioDataset._nbytes(v) for v in obj)
        return 0

    def sample_features(self, *features: Any, max: int = 10000, batch_size: int = 128, max_size: int | str | None = None,
                        reduction: Callable[[List[Any]], Any] | None = None, cache: str | Path | None = None, **loader_kwargs):
        """loads given features on random items of the dataset, without the transforms of the dataset. Only requested
        features are loaded (online datasets only compute these ones).

        Args:
            *features (str | AcidsDatasetFeature | Transform | Callable): features to sample, each giving an output entry :
                - names (str) of recorded features (pre-processed or computed by the dataset), or "waveform" for audio ;
                - feature objects (AcidsDatasetFeature), computed on loaded audio batches (entry : their feature_name) ;
                - augmentations (Transform, or any callable), applied to loaded audio batches, giving augmented audio
                    (entry : their name if any, their lower-cased class name otherwise).
                Objects are computed in the main process, on the device of the feature if any. Batches are the entry
                for a single feature, and {entry: batch} dicts otherwise.
            max (int, optional): maximum number of loaded batches. Defaults to 10000.
            batch_size (int, optional): batch size of the inner data loader. Defaults to 128.
            max_size (int | str | None, optional): maximum size of loaded data, in bytes or as a string (e.g. "1G", "500M") ;
                loading stops before exceeding it. None for no limit. Defaults to None.
            reduction (Callable | None, optional): called on the list of batches, and its output returned (e.g. a mean).
                Defaults to None (the list of batches is returned).
            cache (str | Path | None, optional): file where loaded batches are saved ; if it exists and was written with
                the same features, max, batch_size and max_size, batches are read from it instead of loaded. Objects are
                compared by class and attributes. Defaults to None.
            **loader_kwargs (Any): additional arguments of torch.utils.data.DataLoader (e.g. num_workers).
        """
        if len(features) == 0:
            raise ValueError("sample_features requires at least one feature")
        entries = self._sample_entries(features)
        cache_key = {'features': [(name, obj if isinstance(obj, str) else self._describe(obj)) for name, obj in entries], 
                     'max': max, 'batch_size': batch_size, 'max_size': max_size}
        batches = None
        if cache is not None and Path(cache).exists():
            cached = torch.load(cache, weights_only=False)
            if isinstance(cached, dict) and cached.get('key') == cache_key:
                batches = cached['batches']
            else:
                logging.warning("cache %s was written with other parameters ; features are loaded again"%cache)
        if batches is None:
            batches = self._load_feature_batches(entries, max, batch_size, self._parse_size(max_size), loader_kwargs)
            if cache is not None:
                Path(cache).parent.mkdir(parents=True, exist_ok=True)
                torch.save({'key': cache_key, 'batches': batches}, cache)
        return batches if reduction is None else reduction(batches)

    @staticmethod
    def _sample_entries(features) -> List[tuple]:
        """(entry name, feature name or object) of the arguments of sample_features ; duplicate names get a suffix."""
        from ..features import AcidsDatasetFeature
        entries, names = [], set()
        for f in features:
            if isinstance(f, str):
                name = f
            elif isinstance(f, AcidsDatasetFeature):
                name = f.feature_name
            elif callable(f):
                name = getattr(f, "name", None) or type(f).__name__.lower()
            else:
                raise TypeError("sample_features takes feature names, feature objects or augmentations, got %r"%(f,))
            base, i = name, 1
            while name in names:
                name, i = "%s_%d"%(base, i), i + 1
            names.add(name)
            entries.append((name, f))
        return entries

    @staticmethod
    def _describe(obj) -> str:
        """description of an object for cache keys : class, and attributes of simple types."""
        simple = (int, float, str, bool, type(None), tuple, list)
        attrs = {k: v for k, v in sorted(vars(obj).items()) if isinstance(v, simple)} if hasattr(obj, "__dict__") else {}
        return "%s(%s)"%(type(obj).__name__, attrs)

    @staticmethod
    def _apply_to_batch(obj, wav):
        """computes a feature object (or applies an augmentation) on a batch of audio (or a list of items)."""
        from ..features import AcidsDatasetFeature
        if isinstance(wav, list):
            return collate_items([AudioDataset._apply_to_batch(obj, w[None])[0] for w in wav])
        wav = torch.as_tensor(wav)
        device = getattr(obj, "device", None)
        if device is not None:
            wav = wav.to(device)
        out = obj.batch_call(wav) if isinstance(obj, AcidsDatasetFeature) else obj(wav)
        return out.cpu() if torch.is_tensor(out) else out

    def _load_feature_batches(self, entries: List[tuple], max_batches: int, batch_size: int, max_bytes: int | None,
                              loader_kwargs: Dict[str, Any]) -> List[Any]:
        recorded = [f for _, f in entries if isinstance(f, str)]
        objects = [(name, f) for name, f in entries if not isinstance(f, str)]
        # objects are computed on loaded audio
        loaded = recorded + (["waveform"] if objects and "waveform" not in recorded else [])
        computed = [n for n in self._computed_features() if n not in loaded]
        dataset = self._without_features(computed)
        dataset._set_worker_outputs("{%s}"%",".join(loaded), None)
        n_items = min(len(dataset), max_batches * batch_size)
        sampler = torch.randperm(len(dataset))[:n_items].tolist()
        loader_kwargs = {'collate_fn': collate_items, **loader_kwargs}
        if loader_kwargs.get('num_workers', 0) > 0 and loader_kwargs.get('multiprocessing_context') is None and dataset.feature_devices - {"cpu"}:
            loader_kwargs['multiprocessing_context'] = "spawn"
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, **loader_kwargs)
        batches, total = [], 0
        for batch in loader:
            if isinstance(batch, list):
                # items that cannot be collated (e.g. whole files of different lengths)
                batch = {k: [item[k] for item in batch] for k in loaded}
            batch = {name: batch[f] if isinstance(f, str) else self._apply_to_batch(f, batch["waveform"]) for name, f in entries}
            batch = batch[entries[0][0]] if len(entries) == 1 else batch
            size = self._nbytes(batch)
            if max_bytes is not None and total + size > max_bytes:
                logging.warning("sample_features : max_size (%d bytes) reached after %d batches"%(max_bytes, len(batches)))
                break
            batches.append(batch)
            total += size
            if len(batches) >= max_batches:
                break
        if len(batches) == 0:
            raise RuntimeError("sample_features : no batch could be loaded (first batch exceeds max_size=%s bytes?)"%max_bytes)
        return batches

    # batch features

    def _computed_features(self) -> Dict[str, Any]:
        """features computed dynamically by the dataset (and not read from storage), by name."""
        return {}

    def _resolve_batch_features(self, features) -> List[Any]:
        """feature objects of batch features, given by name, as objects, or as "auto" (computed features on accelerators)."""
        computed = self._computed_features()
        if features is None:
            return []
        if isinstance(features, str) and features == "auto":
            return [f for f in computed.values() if getattr(f, "device", None) is not None and torch.device(f.device).type != "cpu"]
        resolved = []
        for f in (features if isinstance(features, (list, tuple)) else [features]):
            if not isinstance(f, str):
                resolved.append(f)
            elif f in computed:
                resolved.append(computed[f])
            elif f in self._stored_feature_names():
                raise ValueError("feature %s is pre-processed, and is read from the dataset ; give a feature object to compute it on batches"%f)
            else:
                raise ValueError("feature %s is not computed by the dataset ; available features : %s"%(f, list(computed.keys())))
        return resolved

    def _stored_feature_names(self) -> List[str]:
        features = self.features
        return list(features.keys()) if isinstance(features, dict) else [getattr(f, "feature_name", None) for f in features or []]

    def _without_features(self, names: List[str]) -> "AudioDataset":
        """returns a copy of the dataset that does not compute given features."""
        import copy
        return copy.copy(self)

    def _set_worker_outputs(self, output_pattern: str, transforms) -> None:
        """sets the output pattern and transforms of a copy of the dataset used in workers."""
        self._output_pattern = output_pattern
        self._transforms = transforms

    def compute_batch_features(self, audio: torch.Tensor, features = None, device = None) -> Dict[str, torch.Tensor]:
        """computes features on a batch of audio (batch, channels, samples), e.g. in a training loop with a custom data 
        loader. features are names of features computed by the dataset or feature objects (by default, every feature 
        computed by the dataset). Returns {name: batched feature}."""
        features = self._resolve_batch_features(list(self._computed_features().keys()) if features is None else features)
        audio = torch.as_tensor(audio)
        if audio.ndim == 2: audio = audio[:, None]
        if device is not None:
            from ..utils import resolve_device
            device = resolve_device(device)
            audio = audio.to(device)
            for f in features:
                if hasattr(f, "to"): f.to(device)
        return {f.feature_name: f.batch_call(audio) for f in features}

    def _check_loader_arguments(self, batch_size: int, loader_kwargs: Dict[str, Any]) -> None:
        """checks data loader arguments before building a loader (e.g. for datasets with items of different sizes)."""
        pass

    # devices

    def _feature_objects(self) -> List[Any]:
        features = self.features
        return list(features.values()) if isinstance(features, dict) else list(features or [])

    @property
    def feature_devices(self) -> set:
        """device types (cpu, cuda, mps...) of features computed by the dataset."""
        devices = set()
        for f in self._feature_objects():
            device = getattr(f, "device", None)
            if device is not None:
                devices.add(torch.device(device).type)
        return devices

    def to(self, device: str | torch.device) -> "AudioDataset":
        """moves features computed by the dataset (e.g. neural embeddings) to device, and returns the dataset."""
        for f in self._feature_objects():
            if hasattr(f, "to"): f.to(device)
        return self

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, obj):
        assert issubclass(type(obj), type(self)), "parent of a dataset must be a subclass of %s"%(type(self))
        self._parent = obj

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

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, transforms):
        self._transforms = transforms


__all__ = ['AudioDataset', 'is_preprocessed_path', 'get_channels_from_path', 'channels_file_path', 'TransformType']
