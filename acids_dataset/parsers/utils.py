import traceback

class FileNotReadException(Exception):
    def __init__(self, file, backend, original_exception):
        self.file = file
        self.backend = backend
        self.exception = original_exception
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return f"FileNotReadException(backend={self.backend.__name__}, file={self.file}, exception={traceback.format_exception(self.exception)})"
    