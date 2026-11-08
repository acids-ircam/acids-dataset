"""Batch features : features computed on whole batches in the main process (e.g. on GPU), rather than item by item
in data loader workers. Workers only load audio (and compute other features) ; batch features are then computed on the
batch, after the transforms of the waveform, or before them if the dataset has features_after_transforms=False (see 
AudioDataset.get_loader and AudioDataset.compute_batch_features)."""
import re
from typing import Any, Dict, List, Optional
import torch

from .utils import _check_channels

_SUBINDEX = re.compile(r"^(\S+)\[([\d\:\-]+)\]$")
AUDIO_KEY = "__audio__"


def _out_key(i: int) -> str:
    return "__out_%d__"%i


def parse_output_pattern(pattern: str):
    """parses an output pattern into its kind ("single", "tuple" or "dict") and its entries, as (source, alias)
    tuples (e.g. "{waveform[0]->x,loudness}" -> ("dict", [("waveform[0]", "x"), ("loudness", "loudness")]))."""
    pattern = pattern.strip()
    if re.match(r"^(\w+)$", pattern):
        return "single", [(pattern, pattern)]
    result = re.match(r"^\{(.+)\}$", pattern)
    if result is not None:
        entries = []
        for item in filter(None, map(str.strip, result.groups()[0].split(","))):
            source, alias = map(str.strip, item.split("->")) if "->" in item else (item, item)
            entries.append((source, alias))
        return "dict", entries
    result = re.match(r"^\(?(.+?)\)?$", pattern)
    entries = [(item, item) for item in filter(None, map(str.strip, result.groups()[0].split(",")))]
    return "tuple", entries


def entry_name(source: str) -> str:
    """base name of an entry (without sub-index and method suffix, e.g. "waveform:audio[0]" -> "waveform")."""
    result = _SUBINDEX.match(source)
    if result is not None:
        source = result.groups()[0]
    return source.split(":")[0]


def entry_subindex(source: str):
    result = _SUBINDEX.match(source)
    if result is None:
        return None
    subidx = result.groups()[1]
    if ":" in subidx:
        return slice(*[int(x) if x != "" else None for x in subidx.split(":")])
    return int(subidx)


def augmented_audio_entry(entries: List[tuple], transforms: List[Any], exclude=()) -> Optional[int]:
    """index of the waveform entry features are computed on : the first plain waveform entry (without sub-index) that
    has a transform, such that features are computed on augmented audio, or the first plain waveform entry if none
    has a transform ; None if there is no plain waveform entry."""
    candidates = [i for i, (source, _) in enumerate(entries) if i not in exclude and source.split(":")[0] == "waveform"]
    for i in candidates:
        if transforms[i] is not None:
            return i
    return candidates[0] if candidates else None


def clean_audio_entry(entries: List[tuple], transforms: List[Any], exclude=()) -> Optional[int]:
    """index of the first plain waveform entry without transform (i.e. loaded audio), None if there is none."""
    for i, (source, _) in enumerate(entries):
        if i not in exclude and source.split(":")[0] == "waveform" and transforms[i] is None:
            return i
    return None


def transforms_per_entry(kind: str, entries: List[tuple], transforms) -> List[Any]:
    """transforms of each entry, following the conventions of output patterns (one transform for single patterns,
    lists for tuples, dicts of aliases for dicts)."""
    if transforms is None:
        return [None] * len(entries)
    if kind == "dict":
        assert isinstance(transforms, dict), "transforms of dict output patterns must be dicts"
        return [transforms.get(alias) for _, alias in entries]
    if kind == "tuple":
        assert isinstance(transforms, (list, tuple)), "transforms of tuple output patterns must be lists"
        return [transforms[i] if i < len(transforms) else None for i in range(len(entries))]
    if isinstance(transforms, (list, tuple)):
        return [transforms[0] if len(transforms) > 0 else None]
    return [transforms]


def _to_device(x, device):
    return x.to(device) if torch.is_tensor(x) and device is not None else x


class BatchFeatureProcessor():
    """computes batch features (and optionally transforms) on batches given by data loader workers.

    Args:
        dataset (AudioDataset): configured dataset.
        features (List[str | AcidsDatasetFeature] | str | None): features computed on batches, by name (features computed by
            the dataset) or as feature objects (that may not belong to the dataset) ; "auto" for features on accelerators.
        device (str | torch.device | None): device of batches, batch features and batch transforms ("auto" for the default
            accelerated device).
        batch_transforms (bool): apply transforms on batches (on device) rather than in workers.
    """

    def __init__(self, dataset, features=None, device=None, batch_transforms: bool = False):
        from ..utils import resolve_device
        self.device = None if device is None else resolve_device(device)
        self.batch_transforms = batch_transforms
        self.after_transforms = dataset.features_after_transforms
        self.features = dataset._resolve_batch_features(features)
        self.kind, self.entries = parse_output_pattern(dataset.output_pattern)
        self.transforms = transforms_per_entry(self.kind, self.entries, dataset.transforms)
        if batch_transforms and self.after_transforms:
            # augmentations are applied on batches : features computed in workers would not see them, and are hence
            # computed on batches
            audio_entry = augmented_audio_entry(self.entries, self.transforms)
            if audio_entry is not None and self.transforms[audio_entry] is not None:
                computed, names = dataset._computed_features(), [f.feature_name for f in self.features]
                for source, _ in self.entries:
                    name = entry_name(source)
                    if name in computed and name not in names:
                        self.features.append(computed[name]); names.append(name)
        names = [f.feature_name for f in self.features]
        self.n_channels = dataset._n_channels
        self.batch_entries = [entry_name(source) in names for source, _ in self.entries]
        # features are computed on the augmented waveform output (after its transforms) if any, and on raw audio otherwise
        # (or always on raw audio if features are computed before transforms)
        batch_entries = {i for i, b in enumerate(self.batch_entries) if b}
        if self.after_transforms:
            self.audio_entry = augmented_audio_entry(self.entries, self.transforms, exclude=batch_entries)
        else:
            self.audio_entry = clean_audio_entry(self.entries, self.transforms, exclude=batch_entries)
        worker_entries = ["%s->%s"%(source, _out_key(i)) for i, (source, _) in enumerate(self.entries) if not self.batch_entries[i]]
        if self.audio_entry is None and self.features:
            worker_entries.append("waveform->%s"%AUDIO_KEY)
        worker_transforms = {} if batch_transforms else {_out_key(i): t for i, t in enumerate(self.transforms)
                                                         if t is not None and not self.batch_entries[i]}
        self.worker_dataset = dataset._without_features(names)
        self.worker_dataset._set_worker_outputs("{%s}"%",".join(worker_entries), worker_transforms)
        if self.device is not None:
            for f in self.features:
                if hasattr(f, "to"): f.to(self.device)

    def _transform(self, i, x):
        t = self.transforms[i]
        if t is None:
            return x
        return t(_check_channels(x, self.n_channels))

    def __call__(self, batch: Dict[str, Any]):
        batch = {k: _to_device(v, self.device) for k, v in batch.items()}
        outs = [None] * len(self.entries)
        for i in range(len(self.entries)):
            if not self.batch_entries[i]:
                outs[i] = self._transform(i, batch[_out_key(i)]) if self.batch_transforms else batch[_out_key(i)]
        if self.features:
            audio = outs[self.audio_entry] if self.audio_entry is not None else batch[AUDIO_KEY]
            computed = {f.feature_name: f.batch_call(audio) for f in self.features}
            for i, (source, _) in enumerate(self.entries):
                if self.batch_entries[i]:
                    value = computed[entry_name(source)]
                    subidx = entry_subindex(source)
                    if subidx is not None:
                        value = value[:, subidx]
                    outs[i] = self._transform(i, value)
        if self.kind == "single":
            return outs[0]
        if self.kind == "tuple":
            return tuple(outs)
        return {alias: outs[i] for i, (_, alias) in enumerate(self.entries)}


class BatchFeatureLoader():
    """data loader whose batches are processed by a BatchFeatureProcessor (see AudioDataset.get_loader). Attributes of
    the underlying torch.utils.data.DataLoader (dataset, sampler, batch_size...) are available."""

    def __init__(self, loader: torch.utils.data.DataLoader, processor: BatchFeatureProcessor):
        self.loader = loader
        self.processor = processor

    def __iter__(self):
        for batch in self.loader:
            yield self.processor(batch)

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        if name in ("loader", "processor"):
            raise AttributeError(name)
        return getattr(self.loader, name)


__all__ = ['BatchFeatureProcessor', 'BatchFeatureLoader', 'parse_output_pattern', 'augmented_audio_entry']
