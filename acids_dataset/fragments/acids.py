import gin
import numpy as np
from . import check_compiled_proto
from .base import AudioFragment
from typing import Literal, Optional, Any
from ..utils import get_backend


RaveExampleClass = check_compiled_proto(__file__)


DTYPE_TO_PRECISION = {
    "int16": RaveExampleClass.Precision.INT16,
    "int32": RaveExampleClass.Precision.INT32,
    "int64": RaveExampleClass.Precision.INT64,
    "float16": RaveExampleClass.Precision.FLOAT16,
    "float32": RaveExampleClass.Precision.FLOAT32,
    "float64": RaveExampleClass.Precision.FLOAT64,
}

PRECISION_TO_DTYPE = {
    RaveExampleClass.Precision.INT16: "int16",
    RaveExampleClass.Precision.INT32: "int32",
    RaveExampleClass.Precision.INT64: "int64",
    RaveExampleClass.Precision.FLOAT16: "float16",
    RaveExampleClass.Precision.FLOAT32: "float32",
    RaveExampleClass.Precision.FLOAT64: "float64",
}


@gin.configurable(module="ad.fragments")
class AcidsFragment(AudioFragment):

    _metadata_exclude_list = []
    ExampleClass = RaveExampleClass
    DTYPE_TO_PRECISION = DTYPE_TO_PRECISION
    PRECISION_TO_DTYPE = PRECISION_TO_DTYPE
    force_array_reshape = True

    def init_from_kwargs(
            self,
            audio_path: Optional[str] = None, 
            audio: Optional[Any] = None,
            sr: Optional[int] = None, 
            start_pos: Optional[float] = None, 
            length: Optional[float] = None,
            file_length: Optional[float] = None,
            bformat: Optional[str] = "int8",
            output_type: Literal["numpy", "torch", "jax"] = "numpy",
            **kwargs
            ) -> None:

        metadata = dict(kwargs)
        #TODO: clean, that, select mandatory fields (make a dataclass?)
        if audio_path is not None: metadata['audio_path'] = str(audio_path)
        if start_pos is not None: metadata['start_pos'] = str(start_pos)
        if bformat is not None: metadata['bformat'] = str(bformat)
        if output_type is not None: metadata['output_type'] = str(output_type)
        if length is not None: metadata['length'] = str(length)
        if file_length is not None: metadata['file_length'] = str(file_length)
        self.set_metadata(metadata)
        if audio is not None:
            self.put_audio("waveform", audio, dtype=bformat, sr=sr)

    @property
    def raw_audio(self):
        return self.get_audio("waveform")
