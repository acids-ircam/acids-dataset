import enum
from absl import logging
from typing import List, Callable
import math
import torch, torchaudio
from dataclasses import dataclass
from pathlib import Path
import gin
from ..features import AcidsDatasetFeature

from .utils import FileNotReadException
from ..utils import loudness, PadMode, pad


class TorchaudioBackend(object):
    @classmethod
    def parse_file(
        cls,
        audio_path, 
        chunk_length, 
        hop_length, 
        pad_mode,
        sr,
        channels,
        bformat,
        loudness_threshold, 
        features, 
        dataset_path, 
        waveform, 
        discard_if_lower_than
    ) -> List[Callable]:
        def _load_file(fragment_class = dict, audio_path=audio_path):
            try:
                wav, orig_sr = torchaudio.load(audio_path)
            except RuntimeError: 
                raise FileNotReadException(audio_path, cls)
            if orig_sr != sr:
                wav = torchaudio.functional.resample(wav, orig_sr, sr)
            if channels is not None:
                if wav.shape[0] > channels:
                    wav = wav[:channels]
                elif wav.shape[0] < channels: 
                    wav = wav[torch.arange(channels)%wav.shape[0]]
            for f in features:
                if hasattr(f, "pre_chunk_hook"): f.pre_chunk_hook(audio_path, wav, sr, dataset_path)
            file_length = wav.shape[-1] / sr
            chunk_length_smp = int(chunk_length * sr)
            hop_length_smp = int(hop_length * sr)
            n_chunks = math.ceil(wav.shape[-1] / hop_length_smp)
            start_pos = [i*hop_length_smp for i in range(n_chunks)]
            wav = [wav[..., start_pos[i]:start_pos[i]+chunk_length_smp] for i in range(n_chunks)]
            fragments = []
            for i in reversed(range(len(wav))):
                if wav[i].shape[-1] != chunk_length_smp:
                    padded_chunk = cls.pad_chunk(wav[i], chunk_length_smp, pad_mode, discard_if_lower_than)
                    if padded_chunk is None: 
                        del wav[i]
                        continue
                    else:
                        wav[i] = padded_chunk
                
                if loudness_threshold is not None:
                    chunk_loudness = loudness(wav[i], sr)
                    if chunk_loudness < loudness_threshold:
                        del wav[i]
                        continue
                fragment_audio_path = str(audio_path)
                if dataset_path is not None:
                    fragment_audio_path = Path(fragment_audio_path).resolve().relative_to(dataset_path.parent)
                current_fragment = fragment_class(
                        audio_path = fragment_audio_path, 
                        audio = wav[i] if waveform else None, 
                        start_pos = start_pos[i] / sr,
                        length = chunk_length,
                        file_length = file_length,
                        bformat = bformat,
                        sr = sr
                    )
                fragments.insert(0, current_fragment)
            return fragments 

        _load_file.__repr__ = lambda: "TorchaudioBackend.file_loader(audio_path=%s)"%(audio_path)
        return [_load_file]

    @classmethod
    def pad_chunk(
        cls, 
        chunk,
        target_size,
        pad_mode, 
        discard_if_lower_than
    ):
        return pad(chunk, target_size, pad_mode, discard_if_lower_than=discard_if_lower_than)        

class FFMPEGBackend(object):
    @classmethod
    def parse_file(
        cls,
        path, 
        chunk_length, 
        hop_length, 
        pad_mode,
        sr,
        channels,
        bformat,
        loudness_threshold 
    ) -> List[Callable]:
        raise NotImplementedError()

    @classmethod
    def pad_chunk(
        cls, 
        chunk,
        target_size,
        pad_mode
    ):
        raise NotImplementedError()


class ImportBackend(enum.Enum):
    TORCHAUDIO = 0
    # FFMPEG = 1

    @property
    def backend(self):
        if self == ImportBackend.TORCHAUDIO:
            return TorchaudioBackend
        # elif self == ImportBackend.FFMPEG:
        #     return FFMPEGBackend
        else:
            raise ValueError('No backend for %s'%self)


@gin.configurable(module="parser")
class RawParser(object):
    def __init__(
            self, 
            audio_path: str,
            chunk_length: int | float, 
            sr: int = 44100, 
            channels: int = 1, 
            hop_length: int | float | None = None,
            overlap: float | None = None, 
            pad_mode: PadMode | str = PadMode.DISCARD,
            import_backend: ImportBackend | str = ImportBackend.TORCHAUDIO,
            bformat: str = "int16", 
            loudness_threshold: float | None = None,
            features: List[AcidsDatasetFeature] | None = None,
            dataset_path: str | None = None,
            waveform: bool = True, 
            discard_if_lower_than: int | float | None = None
        ):
        self.audio_path = audio_path 
        self.dataset_path = dataset_path
        assert Path(self.audio_path).exists(), f"{self.audio_path} does not seem to exist. Please provide a valid file"
        if chunk_length is None:
            raise ValueError('RawParser needs a chunk length.')
        if isinstance(chunk_length, int):
            chunk_length = chunk_length / sr
        self.chunk_length = float(chunk_length)
        assert self.chunk_length > 0, "chunk_length must be positive"
        self.sr = int(sr)
        assert self.sr > 0, "sr must be positive" 
        if (hop_length is not None) and (overlap is None):
            assert hop_length > 0, "hop_length must be positive"
            if isinstance(hop_length, int):
                hop_length = hop_length / self.sr
            self.hop_length = hop_length
        elif (hop_length is None) and (overlap is not None):
            self.hop_length = (1 - overlap) * chunk_length
        elif (hop_length is None) and (overlap is None):
            self.hop_length = chunk_length
        else:
            raise ValueError("overlap and overlap_ratio must not be given at the same time.")
        if isinstance(pad_mode, str):
            pad_mode = getattr(PadMode, pad_mode.upper())
        self.pad_mode = pad_mode
        self.bformat = bformat
        if isinstance(import_backend, str):
            import_backend = getattr(ImportBackend, import_backend.upper())
        self.import_backend = import_backend
        self.loudness_threshold = loudness_threshold
        self.channels = channels
        self.waveform = waveform
        self.discard_if_lower_than = discard_if_lower_than
        self._parse_features(features or [])

    @property
    def chunk_length_smp(self):
        return int(self.chunk_length * self.sr)
    
    @property
    def hop_length_smp(self):
        return int(self.hop_length * self.sr)

    def _parse_features(self, features) -> None:
        self._features = []
        for f in features:
            if getattr(f, "pre_chunk_hook", None):
                self._features.append(f)
    
    def get_metadata(self):
        return {
            'chunk_length': self.chunk_length, 
            'hop_length': self.hop_length, 
            'import_backend': self.import_backend.value,
            'pad_mode': self.pad_mode.value, 
            'format': self.bformat,
            'sr': self.sr, 
            'channels': self.channels,
            'loudness_threshold': self.loudness_threshold
        }

    def __iter__(self):
        parsing_method = self.import_backend.backend.parse_file
        return iter(parsing_method(
            audio_path=self.audio_path, 
            chunk_length=self.chunk_length, 
            hop_length=self.hop_length, 
            pad_mode=self.pad_mode,
            sr=self.sr,
            channels=self.channels,
            bformat=self.bformat,
            loudness_threshold=self.loudness_threshold, 
            features = self._features, 
            dataset_path=self.dataset_path, 
            waveform=self.waveform,
            discard_if_lower_than = self.discard_if_lower_than
        ))


__all__ = ['RawParser', "PadMode", "ImportBackend", "FileNotReadException"]




