import os, gin

RAISE_EXC_IF_WRITER_ERROR = bool(os.environ.get('ACIDSTRANSFORMS_RAISE_EXC_IF_WRITER_ERROR', True))

from .utils import *
from .lmdb_writer import LMDBWriter, LMDBLoader

@gin.configurable()
def get_writer_class(writer_class = LMDBWriter, filters=[], exclude=[]):
    with gin.unlock_config():
        gin.bind_parameter(f"{writer_class.__name__}.filters", gin.get_bindings(writer_class.__name__).get('filters',[]) + checklist(filters))
        gin.bind_parameter(f"{writer_class.__name__}.exclude", gin.get_bindings(writer_class.__name__).get('exclude', []) + checklist(exclude)) 
    return writer_class
