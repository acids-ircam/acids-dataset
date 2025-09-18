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

def _from_fragment(fragment, item):
    item = item.split(':')
    if len(item) == 1:
        item, method_type = item[0], None
    elif len(item) == 2:
        item, method_type = item
    else:
        raise ValueError('invalid item found : %s'%item)
    return getattr(fragment, fragment_interfaces[method_type])(item)

def _outs_from_pattern(fragment, output_pattern):
    re_result = re.match(r'^(\w+)$', output_pattern)
    if re_result is not None:
        return _from_fragment(fragment, output_pattern)
    re_result = re.match(r'^\(?([\w\,\: \[\]\d]+)\)?$', output_pattern)
    if re_result is not None:
        items = list(filter(lambda x: x != "", re_result.groups()[0].split(',')))
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
            out = _from_fragment(fragment, entry.replace(" ", ""))
            if subidx is not None: 
                out = out[subidx]
            item_parsed.append(out)
        return tuple(item_parsed)
    re_result = re.match(r'^\{([\w\,\: \[\]\d\-\>]+)\}$', output_pattern)
    if re_result is not None:
        items = list(filter(lambda x: x != "", re_result.groups()[0].split(',')))
        items_parsed = {}
        for i in items:
            entry = i
            alias = i 
            out = None
            
            if ('->' in i):
                if len(i.split('->')) == 2:
                    entry, alias = i.split('->')
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
            out = _from_fragment(fragment, entry.replace(" ", ""))
            if subidx is not None: 
                out = out[subidx]
            items_parsed[alias] = out
        return items_parsed
    raise ValueError('Could not parse output_pattern : %s'%output_pattern)

def _check_channels(out, n_channels = None):
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

@gin.configurable(module="data")        
def split_dataset(dataset, **parts):
    raise NotImplementedError

@gin.configurable(module="data")
def get_data_loader(dataset, **kwargs):
    return torch.utils.data.DataLoader(dataset, **kwargs)