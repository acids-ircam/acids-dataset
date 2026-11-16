import gin.torch
from torch.utils.data import Sampler, BatchSampler
from ..writers.utils import KeyIterator, FeatureHashComponent
import torch
import re
from math import ceil


fragment_interfaces = {
    'audio': 'get_audio', 
    'array': 'get_array', 
    "buffer": 'get_buffer',
    None: 'get'}

def is_multi_index(index) -> bool:
    """returns True for indices selecting several items (slices, sequences or 1-d arrays of integers)."""
    if isinstance(index, slice):
        return True
    if isinstance(index, (list, tuple)):
        return True
    if torch.is_tensor(index) or hasattr(index, "__array__"):
        return getattr(index, "ndim", 0) > 0
    return False

def multi_index_to_list(index, length: int) -> list:
    """converts a multi index to a list of indices."""
    if isinstance(index, slice):
        return list(range(*index.indices(length)))
    if torch.is_tensor(index):
        index = index.tolist()
    elif hasattr(index, "__array__"):
        import numpy as np
        index = np.asarray(index).tolist()
    return list(index)

def collate_items(items: list):
    """collates items as a batch (as torch DataLoader), or returns them as a list if they cannot be collated."""
    if len(items) == 0:
        return []
    try:
        return torch.utils.data.default_collate(items)
    except (RuntimeError, TypeError):
        return items

def _from_fragment(fragment, item, index=None):
    # index of the item in the dataset (e.g. to record losses of items, see acids_dataset.sampling.LossTracker)
    if item == "index":
        if index is None:
            raise KeyError("index is not available")
        return torch.tensor(index)
    item = item.split(':')
    if len(item) == 1:
        item, method_type = item[0], None
    elif len(item) == 2:
        item, method_type = item
    else:
        raise ValueError('invalid item found : %s'%item)
    return getattr(fragment, fragment_interfaces[method_type])(item)

def _outs_from_pattern(fragment, output_pattern, index=None):
    re_result = re.match(r'^(\w+)$', output_pattern)
    if re_result is not None:
        return _from_fragment(fragment, output_pattern, index=index)
    re_result = re.match(r'^\(?([\w\,\: \[\]\d]+)\)?$', output_pattern)
    if re_result is not None:
        items = list(filter(lambda x: x != "", map(str.strip, re_result.groups()[0].split(','))))
        item_parsed = []
        for i in items:
            entry = i
            re_result = re.match(r"^(\S+)\[([\d\:]+)\]$", entry)
            if re_result is not None: 
                entry, subidx = re_result.groups()
                if ":" in subidx: 
                    subidx = slice(*map(lambda x: int(x) if x != "" else None, subidx.split(":")))
                else:
                    subidx = int(subidx)
            else:
                subidx = None
            out = _from_fragment(fragment, entry.replace(" ", ""), index=index)
            if subidx is not None: 
                out = out[subidx]
            item_parsed.append(out)
        return tuple(item_parsed)
    re_result = re.match(r'^\{([\w\,\: \[\]\d\-\>]+)\}$', output_pattern)
    if re_result is not None:
        items = list(filter(lambda x: x != "", map(str.strip, re_result.groups()[0].split(','))))
        items_parsed = {}
        for i in items:
            entry = i
            alias = i 
            out = None
            
            if ('->' in i):
                if len(i.split('->')) == 2:
                    entry, alias = map(str.strip, i.split('->'))
                else:
                    raise ValueError('output_pattern %s is wrong: only one alias (aka ->) is supported'%output_pattern)
            re_result = re.match(r"^(\S+)\[([\d\:]+)\]$", entry)
            if re_result is not None: 
                entry, subidx = re_result.groups()
                if ":" in subidx: 
                    subidx = slice(*map(lambda x: int(x) if x != "" else None, subidx.split(":")))
                else:
                    subidx = int(subidx)
            else:
                subidx = None
            out = _from_fragment(fragment, entry.replace(" ", ""), index=index)
            if subidx is not None: 
                out = out[subidx]
            items_parsed[alias] = out
        return items_parsed
    raise ValueError('Could not parse output_pattern : %s'%output_pattern)

def _check_channels(out, n_channels = None):
    # scalars (e.g. loudness of a chunk) have no channels
    if not hasattr(out, "ndim") or out.ndim == 0:
        return out
    if out.ndim == 1: 
        out = out.unsqueeze(0)
    if n_channels is None: 
        return out
    if n_channels is not None:
        if n_channels < out.shape[-2]:
            out= out[..., :n_channels, :]
        elif n_channels > out.shape[-2]:
            out = torch.cat([out] * ceil(n_channels / out.shape[-2]), -2)
            out = out[..., :n_channels, :]
    return out

def _transform_outputs(outs, transforms, n_channels=None):
    if transforms is None: 
        return outs
    # if isinstance(transforms, (tuple, list)):
    #     if len(transforms) == 0: return outs
    #     if len(outs) == 1 and not isinstance(transforms[0], list):
    #         transforms = [transforms]
    if isinstance(outs, (list, tuple)):
        outs = list(outs)
        assert isinstance(transforms, (tuple, list))
        for i, t in enumerate(transforms):
            outs[i] = t(_check_channels(outs[i], n_channels))
        return tuple(outs)
    elif isinstance(outs, dict):
        assert isinstance(transforms, dict)
        for k, t in transforms.items():
            assert k in outs
            outs[k] = t(_check_channels(outs[k], n_channels))
    else:
        if isinstance(transforms, (list, tuple)):
            if len(transforms) > 0:
                outs = transforms[0](_check_channels(outs, n_channels))
        else:
            if transforms is not None:
                outs = transforms(_check_channels(outs, n_channels))
    return outs

_FLOAT_DTYPE_ALIASES = {
    'float': torch.float32, 'float32': torch.float32, 'single': torch.float32,
    'double': torch.float64, 'float64': torch.float64,
    'half': torch.float16, 'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}

def parse_float_dtype(dtype) -> torch.dtype | None:
    """parses a floating point dtype, given as a torch.dtype, a numpy dtype, or a string (e.g. "float32", "torch.float16",
    "half"). None is kept as None."""
    if dtype is None or isinstance(dtype, torch.dtype):
        parsed = dtype
    elif isinstance(dtype, str):
        name = dtype.lower().removeprefix("torch.")
        if name not in _FLOAT_DTYPE_ALIASES:
            raise ValueError("unknown float dtype %s (valid : %s)"%(dtype, ", ".join(_FLOAT_DTYPE_ALIASES)))
        parsed = _FLOAT_DTYPE_ALIASES[name]
    else:
        import numpy as np
        parsed = torch.from_numpy(np.zeros(0, dtype=dtype)).dtype
    if parsed is not None and not parsed.is_floating_point:
        raise ValueError("float_dtype must be a floating point dtype, got %s"%dtype)
    return parsed

def _cast_floats(outs, dtype: torch.dtype | None):
    """casts floating point tensors / arrays of (possibly nested) outputs to dtype ; other outputs are unchanged."""
    if dtype is None:
        return outs
    if torch.is_tensor(outs):
        return outs.to(dtype) if outs.is_floating_point() and outs.dtype != dtype else outs
    if isinstance(outs, dict):
        return {k: _cast_floats(v, dtype) for k, v in outs.items()}
    if isinstance(outs, (list, tuple)):
        return type(outs)(_cast_floats(v, dtype) for v in outs)
    import numpy as np
    if isinstance(outs, np.ndarray) and np.issubdtype(outs.dtype, np.floating):
        if dtype == torch.bfloat16:
            # numpy has no bfloat16 : arrays are converted to tensors
            return torch.from_numpy(outs).to(dtype)
        np_dtype = torch.empty(0, dtype=dtype).numpy().dtype
        return outs if outs.dtype == np_dtype else outs.astype(np_dtype)
    return outs

@gin.configurable(module="ad.data")
def split_dataset(dataset, **parts):
    raise NotImplementedError

@gin.configurable(module="ad.data")
def get_data_loader(dataset, **kwargs):
    return torch.utils.data.DataLoader(dataset, **kwargs)