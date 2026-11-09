
import dill
import torch
import numpy as np
from typing import Any, Optional, Callable, Literal
from types import MethodType
from .utils import dict_from_buffer, dict_to_buffer
from ..utils import get_backend

ArrayType = np.ndarray
if get_backend("torch"):
    ArrayType = ArrayType | get_backend("torch").Tensor

DEFAULT_OUTPUT_TYPE = "numpy"


class AudioFragment(object):
    force_array_reshape = True
    def __init__(self):
        """
        Implements base functions for audio fragments, that are : 
            - encoding / decoding buffers and metadata
            - automatically parsing metadata as attributes
            - base accessing functions for parsers and datasets (raw_audio, etc.)
        """
    
    def __init__(
            self, 
            byte_string: Optional[str] = None, 
            output_type: Literal["numpy", "torch", "jax"] = DEFAULT_OUTPUT_TYPE,
            **kwargs):
        self.__dict__["_setters"] = {}
        self.__dict__["_getters"] = {}
        self.__dict__["_deletters"] = {}
        if byte_string is not None: 
            self.init_from_byte_string(byte_string, output_type=output_type)
        else:
            self.ae = self.ExampleClass()
            self.set_metadata({})
            self.init_from_kwargs(**kwargs, output_type=output_type)
        if getattr(self, "ExampleClass", None) is None:
            raise TypeError("Class %s has no bounded protoclass ; when overloading AudioFragment class, please provide ExampleClass class attribute")
        self._parse_properties()

    def init_from_byte_string(self, bs, output_type=DEFAULT_OUTPUT_TYPE):
        try:
            self.ae = self.ExampleClass.FromString(bs)    
            self.output_type = output_type
        except Exception as e: 
            raise e

    def _record_property_from_metadata(self, attribute):
        def _property_getter_fn(self, key=attribute):
            return self.get_metadata()[key]
        def _property_setter_fn(self, obj, key=attribute):
            self.update_metadata(**{key: obj})
        def _property_del_fn(self, key=attribute):
            self.delete_from_metadata(key)
        self._getters[attribute] = MethodType(_property_getter_fn, self)
        self._setters[attribute] = MethodType(_property_setter_fn, self)
        self._deletters[attribute] = MethodType(_property_del_fn, self)

    def _parse_properties(self):
        metadata_keys = list(self.get_metadata().keys())
        for m in metadata_keys:
            if m not in self._metadata_exclude_list:
                self._record_property_from_metadata(m)

    def serialize(self):
        return self.ae.SerializeToString()


    def __getattr__(self, name: str) -> Any:
        if "_getters" not in self.__dict__:
            raise RuntimeError('AudioFragment class not initialized ; did you call super(...).__init__()? ')
        if name in self._getters: 
            return self._getters[name]()
        else:
            return super(AudioFragment, self).__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if "_getters" not in self.__dict__:
            raise RuntimeError('AudioFragment class not initialized ; did you call super(...).__init__()? ')
        if name in self._setters: 
            self._setters[name](value)
        else:
            super(AudioFragment, self).__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if "_deletters" not in self.__dict__:
            raise RuntimeError('AudioFragment class not initialized ; did you call super(...).__init__()? ')
        if name in self._deletters: 
            self._deletters[name]()
        else:
            super(AudioFragment, self).__delattr__(name)

    # buffers classes

    def get(self, key: str):
        buffer = self.get_buffer(key)
        if buffer.unpickler != b'':
            unpickler = dill.loads(buffer.unpickler)
            return unpickler(buffer.data)
        elif getattr(self, "PRECISION_TO_DTYPE", {}).get(buffer.precision) in ["int32", "int64"]:
            # audio is only stored as int16 or float ; other integer buffers are feature arrays
            return self.get_array(key)
        else:
            return self.get_audio(key)


    def get_buffer(self, key: str):
        if key not in self.ae.buffers:
            raise KeyError(f"key '{key}' not available")
        buf = self.ae.buffers[key]

        return buf

    def get_data(self, key: str):
        buf = self.get_buffer(key)
        data = buf.data
        if hasattr(buf, "unpickler"):
            if buf.unpickler != b'':
                data = dill.loads(buf.unpickler)(data) 
        return data

    def get_array(self, key: str, dtype = None, force_dtype: bool = True, conversion_hook: Callable | None = None):
        if not hasattr(self, "PRECISION_TO_DTYPE"):
            raise RuntimeError("AudioFragment.put_array(..) requires PRECISION_TO_DTYPE class property.")

        buf = self.get_buffer(key)
        dtype_decode = getattr(self, "PRECISION_TO_DTYPE")[buf.precision]

        array = np.frombuffer(
            buf.data,
            dtype=dtype_decode
        ).copy()
        try:
            array = array.reshape(buf.shape)
        except Exception as e: 
            if self.force_array_reshape: 
                raise e
            else:
                pass
        
        dtype = dtype or array.dtype
        if self.output_type == "numpy":
            array = array.astype(dtype)
        elif self.output_type == "jax":
            jnp = get_backend('jax')
            array = jnp.array(array)
        elif self.output_type == "torch":
            torch = get_backend('torch')
            array = torch.from_numpy(array)
            if force_dtype:
                array = array.to(getattr(torch, dtype.name))
        else:
            raise ValueError(f"Output type {self.output_type} not available")

        if conversion_hook: 
            array = conversion_hook(array)

        return array

    def get_audio(self, key: str):
        if not hasattr(self, "output_type"):
            raise KeyError('cannot perform get_audio : fragment does not have output_type field')
        # bformat = getattr(self, "bformat", "int16")

        buf = self.get_buffer(key)
        dtype_decode = getattr(self, "PRECISION_TO_DTYPE")[buf.precision]

        if self.output_type == "numpy":
            dtype = np.float64
        elif self.output_type == "torch":
            dtype = torch.float32
        if dtype_decode == "int16":
            hook = lambda x: x / (2**15 - 1)
        elif dtype_decode in ["int32", "int64"]:
            raise NotImplementedError()
        else:
            hook = lambda x: x
        array = self.get_array(key, dtype=dtype, force_dtype=False, conversion_hook=hook)
        return array

    def put_buffer(self, key: str, b: bytes, shape: list, sr: int | None = None, unpickler: Callable | None = None):
        buffer = self.ae.buffers[key]
        buffer.data = b
        if sr is not None:
            buffer.sampling_rate = sr
        if shape is not None:
            buffer.shape.extend(shape)
        buffer.unpickler = dill.dumps(unpickler)

    def put_array(self, key: str, array: ArrayType, dtype: np.dtype | str | None = None, sr: int | None = None):
        if not hasattr(self, "DTYPE_TO_PRECISION"):
            raise RuntimeError("AudioFragment.put_array(..) requires DTYPE_TO_PRECISION class property.")
        array = self._safe_torch_to_numpy(array)
        dtype = np.dtype(dtype or array.dtype)
        buffer = self.ae.buffers[key]
        buffer.data = np.asarray(array).astype(dtype).tobytes()
        if sr is not None:
            buffer.sampling_rate = sr
        for _ in range(len(buffer.shape)):
            buffer.shape.pop()
        buffer.shape.extend(array.shape)
        buffer.precision = getattr(self, "DTYPE_TO_PRECISION")[dtype.name]
    
    def put_audio(self, key: str, array: ArrayType, dtype = np.dtype | str | None, sr: int | None = None): 
        array = self._safe_torch_to_numpy(array)
        dtype = np.dtype(dtype or self.bformat)
        if np.issubdtype(dtype, np.integer):
            array = (array * (2**15 - 1)).astype(dtype)
        self.put_array(key, array, dtype=dtype, sr=sr)

    # metadata classes
    
    def set_metadata(self, metadata: dict):
        meta_buffer = self.ae.buffers["metadata"]
        meta_buffer.data = dict_to_buffer(metadata)

    def update_metadata(self, **kwargs):
        meta_buffer = self.ae.buffers["metadata"]
        meta_buffer = dict_from_buffer(meta_buffer)
        meta_buffer.update(kwargs)
        self.ae.buffers["metadata"].data = dict_to_buffer(meta_buffer)

    def delete_from_metadata(self, *args):
        meta_buffer = self.ae.buffers["metadata"]
        for key in args:
            if key in meta_buffer: del meta_buffer[key]
        meta_buffer.data = dict_to_buffer(meta_buffer)

    def get_metadata(self):
        buf = self.ae.buffers["metadata"]
        if not buf.ByteSize():
            return {}
        else:
            try:
                return dict_from_buffer(buf)
            except Exception as e: 
                print(e)

    def _safe_torch_to_numpy(self, tensor):
        torch = get_backend("torch")
        if not torch:
            return tensor
        if torch.is_tensor(tensor):
            return tensor.cpu().numpy()
        else:
            return tensor
    
    def __str__(self) -> str:
        repr = []
        repr.append("%s("%(type(self).__name__))
        for key in self.ae.buffers:
            if key == "metadata":
                repr.append(str(self.get_metadata()))
            else:
                array = self.get(key)
                repr.append(f"\t{key}[{array.dtype}] {array.shape},")
        repr.append(")")
        return "\n".join(repr)

    def __bytes__(self) -> str:
        return self.ae.SerializeToString()
    
    def as_dict(self):
        return {k: self.get(k) for k in self.ae.buffers}

    def has_buffer(self, name):
        return name in self.ae.buffers

    def delete_buffer(self, name):
        if name in self.ae.buffers:
            del self.ae.buffers[name]

    def has_metadata(self, name):
        return name in dict_from_buffer(self.ae.buffers["metadata"])

    @property 
    def buffers(self):
        return dict(self.ae.buffers)

    @property
    def description(self):
        desc = f"{type(self).__name__}(\n"
        for k, v in self.buffers.items():
            if k == "metadata":
                metadata = self.get_metadata()
                metadata_desc = f"\n\t\t".join(f"{k}: {v}" for k, v in metadata.items())
                desc += f"\t{k}: \n\t\t{metadata_desc}\n"
            else:
                buffer_desc = f"\t{k}:\n\t\tshape: {v.shape}\n\t\tsampling_rate: {v.sampling_rate}\n"
                desc += buffer_desc
        desc += ")"
        return desc


