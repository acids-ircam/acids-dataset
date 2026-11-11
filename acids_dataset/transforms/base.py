import numpy as np
from collections import UserList
from typing import List
import random
import gin, os
import inspect
import numpy as np
import torch
from enum import Enum
from ..utils import get_subclasses_from_package, generate_config_from_obj


gin_config_pattern = """
%s:
\t%s

ad.transforms.parse_transform:
    transform = @ad.transforms.%s()
"""


def check_transform_configs(module, path):
    transform_class = getattr(module, "Transform")
    transform_subclasses = get_subclasses_from_package(module, transform_class)
    os.makedirs(path, exist_ok=True)
    for transform in transform_subclasses:
        if transform == transform_class: continue
        gin_config_name = transform.__name__.lower() + ".gin"
        gin_config_path = (path / gin_config_name).resolve()
        if not gin_config_path.exists():
            transform.write_gin_config(gin_config_path)
        

class AcidsTransformException(Exception):
    pass

class TransformInput(Enum): 
    none = 0
    numpy = 1
    torch = 2

default_cast_table = {
    (TransformInput.numpy, TransformInput.numpy): lambda x: x,
    (TransformInput.numpy, TransformInput.torch): lambda x: torch.from_numpy(x).float(),
    # tensors may be on accelerators (e.g. batches on GPU)
    (TransformInput.torch, TransformInput.numpy): lambda x: x.detach().cpu().numpy(), 
    (TransformInput.torch, TransformInput.torch): lambda x: x
}


gin_config_pattern = """
{{NAME}}:
\t{{ARGS}}

ad.transforms.parse_transform:
    transform = @ad.transforms.{{NAME}}
"""

def check_transform_configs(module, path):
    if not path.exists():
        transform_class = getattr(module, "Transform")
        transform_subclasses = get_subclasses_from_package(module, transform_class)
        os.makedirs(path, exist_ok=True)
        for transform in transform_subclasses:
            if transform == transform_class: continue
            gin_config_name = transform.__name__.lower() + ".gin"
            gin_config_path = (path / gin_config_name).resolve()
            if not gin_config_path.exists():
                generate_config_from_obj(transform, gin_config_path, gin_config_pattern)
            

class Transform():
    # deterministic transforms set allow_random to False : they then cannot be given a probability p
    allow_random: bool = True
    input_types = TransformInput
    takes_as_input = TransformInput.none
    cast_table = default_cast_table
    dont_export_to_gin_config = ["self", "name", "args", "kwargs"]
    def __init__(self,
                 sr: int | None = None, 
                 name: str | None = None, 
                 p: float | None = None, 
                 rand_batchwise: bool = True) -> None:
        """Abstract class for every transform

        Args:
            sr (int | None, optional): Input sampling rate. Defaults to None.
            name (str | None, optional): Optional transform name. Defaults to None.
            p (float | None, optional): Optional probability (must allow random ; deterministic transforms only accept
                None or 1.0, i.e. always applied). Defaults to None.
            rand_batchwise (bool, optional): Is randomness batchwise or indexwise. Defaults to True.
        """
        if not self.allow_random:
            if (p is not None) and (float(p) != 1.0):
                raise AcidsTransformException("%s is not a probabilistic transform : p must be None or 1.0, got %s"%(type(self).__name__, p))
            p = None  # always applied
        self.sr = sr
        self.name = name
        self.p = p
        self.rand_batchwise = rand_batchwise
        self.rng = np.random.default_rng(12345)

    @classmethod
    def init_signature(cls):
        return dict(inspect.signature(cls.__init__).parameters)


    @classmethod
    def write_gin_config(cls, config_path):
        gin_name = f"transforms.{cls.__name__}"
        gin_args = []
        transform_args = cls.init_signature()
        for param_name, param in transform_args.items():
            if param_name in cls.dont_export_to_gin_config: continue
            if param_name == "sr": 
                gin_args.append("sr = %SAMPLE_RATE")
            else:
                default = param._default
                if (param._default == inspect._empty): 
                    continue
                if isinstance(param._default, str): 
                    default = f"\"{default}\""
                gin_args.append(f"{param_name} = {default}")
        gin_args = "\n\t".join(gin_args)
        gin_out = gin_config_pattern%(gin_name, gin_args, cls.__name__)
        with open(config_path, "w+") as f: 
            f.write(gin_out)


    def type_hash(self, data):
        if isinstance(data, np.ndarray):
            return self.input_types.numpy
        elif torch.is_tensor(data):
            return self.input_types.torch
        else: 
            return self.input_types.none

    def _parse_arg(self, arg):
        if self.takes_as_input == self.input_types.none:
            return arg
        else:
            return self.cast_table[self.type_hash(arg), self.takes_as_input](arg)

    def _parse_output(self, output, input):
        if isinstance(output, list):
            return [self.cast_table[(self.type_hash(output[i]), self.type_hash(input[i]))](output[i]) for i in range(len(output))]
        out = self.cast_table[(self.type_hash(output), self.type_hash(input))](output)
        # outputs are given back on the device of inputs
        if torch.is_tensor(out) and torch.is_tensor(input) and out.device != input.device:
            out = out.to(input.device)
        if self.type_hash(out) == self.input_types.torch:
            out = out.to(input)
        return out

    def apply(self, x: np.ndarray):
        return x

    def nn_module(self, **kwargs) -> torch.nn.Module:
        """TorchScript-compatible torch.nn.Module computing the transform (forward), deterministically (the probability p
        is ignored), with an invert method for invertible transforms. Raises a NotImplementedError for transforms without
        module (random augmentations, numpy-based transforms)."""
        raise NotImplementedError("%s has no nn_module"%type(self).__name__)

    def apply_random(self, x):
        if not self.allow_random:
            raise AcidsTransformException("transform %s does not allow random."%type(self).__name__)
        if self.rand_batchwise:
            if torch.is_tensor(x):
                rnm = torch.rand((x.shape[0],) + (1,) * (x.ndim - 1)).expand_as(x)
                x = torch.where(rnm < self.p, self.apply(x), x)
            else:
                rnm = np.broadcast_to(self.rng.uniform(size=(x.shape[0],) + (1,) * (x.ndim - 1)), x.shape)
                x = np.where(rnm < self.p, self.apply(x), x) 
        else:
            if random.random() < self.p:
                x = self.apply(x)
        return x

    def __call__(self, *args, _force_transform: bool = False, **kwargs):
        data_in = args[0]
        args = tuple(map(self._parse_arg, args))
        if (self.p is None) or (not self.allow_random) or (_force_transform):
            out = self.apply(*args, **kwargs)
        else:
            out = self.apply_random(*args, **kwargs)
        out = self._parse_output(out, data_in)
        return out


@gin.configurable(module="ad.transforms")
class Compose(UserList):
    def __init__(self, *args: Transform, transforms: List[Transform] | None = None):
        """Compose sequentially applies conteined transforms. Subclass of UserLit, such that most methods like append, extend,
        etc, are handled.

        Args:
            *args (Transform): transforms to apply.
            transforms (List[Transform], optional): transforms to apply after *args (keyword form, used by gin configs).
        """
        transforms = list(args) + list(transforms or [])
        for i, t in enumerate(transforms):
            assert isinstance(t, Transform), "got wrong type for transform #%d : %s"%(i, type(t))
        super().__init__(transforms)

    def apply(self, x):
        for elm in self: 
            x = elm.apply(x)
        return x

    def nn_module(self, **kwargs) -> torch.nn.Module:
        """module applying the modules of contained transforms sequentially (see Transform.nn_module)."""
        from .nn_modules import ComposeModule
        return ComposeModule([t.nn_module(**kwargs) for t in self])

    def __call__(self, x, **kwargs):
        for elm in self:
            x = elm(x, **kwargs)
        return x


