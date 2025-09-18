import os
import random
import pickle
import shutil
from pathlib import Path
from typing import Any, Callable
from .base import AcidsDatasetFeature, FeatureException, FileHash
from ..utils import load_file, checklist, get_random_hash
from ..transforms.pitch import BasicPitch
import gin
import torch
import pretty_midi
import tempfile

MIDI_EXTS = [".mid", ".midi"]

class FeatureException(Exception):
    pass

def get_midi_from_folder(folder):
    if not os.path.exists(folder):
        return []
    local_files = os.listdir(folder)
    midi_files = list(filter(lambda x: os.path.splitext(x)[1].lower() in MIDI_EXTS, local_files))
    midi_files = [folder / x for x in midi_files]
    return midi_files

def get_midi_from_candidates(midi_paths, original_name=None):
    midi_paths = list(filter(lambda x: x.exists(), midi_paths))
    if len(midi_paths) == 0:
        return None
    elif len(midi_paths) == 1:
        return midi_paths[0]
    else:
        if original_name is not None:
            for f in midi_paths:
                if os.path.splitext(os.path.basename(f))[0] == original_name:
                    return Path(f)
        else:
            return midi_paths[random.randrange(0, len(midi_paths))]

def find_candidates(folder, original_name=None):
    midi_files = get_midi_from_folder(folder)
    midi_file = get_midi_from_candidates(midi_files, original_name=original_name)
    return midi_file
    
def get_midis_for_file(filepath, searchable_paths = ["./"]):
    filepath = Path(filepath)
    candidates = []
    for search_path in searchable_paths:
        midi_paths = get_midi_from_folder(filepath.parent / search_path)
        midi_paths = list(filter(lambda x: x.stem == filepath.stem, midi_paths))
        candidates.extend(midi_paths)
    return candidates


def crop_midi(midi_data, tstart, tend, length): 
    out_notes = []
    tstart = tstart if tstart is not None else 0.
    tend = tend if tend is not None else length
    for note in midi_data.instruments[0].notes:
        if note.end > tstart and note.start < tend:
            note.start = max(0, note.start - tstart)
            note.end = min(note.end - tstart, length)
            out_notes.append(note)
    midi_data.instruments[0].notes = out_notes
    midi_data.adjust_times([0, length], [0, length])
    return midi_data


@gin.configurable(module="features", denylist=AcidsDatasetFeature.denylist)
class AfterMIDI(AcidsDatasetFeature):
    # dictionary of BasicPitch instances, referenced by devices, shared across instances.
    bp = {} 
    feature_name = "midi"
    searchable_midi_paths = [".", "../midi", "./midi", "../MIDI", "./MIDI"]

    def __init__(self, 
                 allow_basic_pitch: bool = True, 
                 relative_midi_path: str | None = None, 
                 **kwargs):
        """acids-dataset adaptation of AFTER implementation pipeline designed by Nils Demerlé."""
        super().__init__(**kwargs)
        self.allow_basic_pitch = allow_basic_pitch
        self.relative_midi_path = relative_midi_path
        self._tmp_midi_folder = Path(tempfile.mkdtemp()).resolve()
        if not self._tmp_midi_folder.exists(): os.makedirs(str(self._tmp_midi_folder))
        self._file_buffer = FileHash()
        os.makedirs(self._tmp_midi_folder, exist_ok=True)

    def __repr__(self):
        return f"AfterMIDI(allow_basic_pitch={self.allow_basic_pitch}, relative_midi_path={self.relative_midi_path}, device={self.device})"

    @classmethod
    def _get_bp_instance(cls, device, sr):
        if cls.bp.get((device, sr)) is None:
            cls.bp[(device, sr)] = BasicPitch(device=device, sr=sr)
        return cls.bp[(device, sr)]

    def _extract_from_basic_pitch(self, audio_data, sr, device=None):
        device = device or self.device
        bp = type(self)._get_bp_instance(device, sr=sr)
        return bp(audio_data)

    def get_midi_path(self, path, relative_midi_path=None):
        relative_midi_path = relative_midi_path or []
        searchable_paths = list(self.searchable_midi_paths)
        searchable_paths.extend(relative_midi_path)
        valid_midi_files = get_midis_for_file(path, searchable_paths=searchable_paths)
        return get_midi_from_candidates(valid_midi_files)

    def _extract_from_file(self, midi_path, sr=None, start=None, end=None, length=None):
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        if (start is not None) or (end is not None):
            assert sr is not None, "sr is required if _extract_from_file is provided a start or a end keyword."
            assert length is not None, "length is required if _extract_from_file is provided a start or a end keyword."
            midi_data = crop_midi(midi_data, start, end, length)
        return midi_data
        
    def _write_midi_buffer(self, path, midi_data):
        midi_path = self._tmp_midi_folder / f"{self._file_buffer.current_id:09d}.mid"
        midi_data.write(str(midi_path))
        self._file_buffer[path] = midi_path

    def get_midi(self, audio_path=None, audio=None, sr=None):
        midi_path = self.get_midi_path(audio_path, relative_midi_path=self.relative_midi_path)
        if midi_path is None:
            if self.allow_basic_pitch:
                midi_data = self._extract_from_basic_pitch(audio, sr, device=self.device)
            else:
                raise FeatureException('midi file for audio %s not found, and allow_basic_pitch is None.')
        else:
            midi_data = self._extract_from_file(midi_path)
        return midi_data

    @property
    def default_feature_name(self):
        return "midi"

    def pre_chunk_hook(self, path, audio, sr, dataset_path):
        midi_data = self.get_midi(audio_path=path, audio=audio, sr=sr)
        self._write_midi_buffer(path, midi_data)

    def close(self):
        if self._tmp_midi_folder.exists():
            shutil.rmtree(self._tmp_midi_folder)

    def from_fragment(self, fragment, write: bool = True):
        audio_path = getattr(fragment, "audio_path", None)
        assert audio_path, f"{type(self)} requires an initialized audio_path metadata from fragment"
        if audio_path in self._file_buffer:
            sr = fragment.get_buffer('waveform').sampling_rate
            metadata = fragment.get_metadata()
            start_pos = float(metadata['start_pos']); length = float(metadata['file_length']); end_pos = start_pos + float(metadata['length'])
            midi_data = self._extract_from_file(self._file_buffer[audio_path], sr, start_pos, end_pos, length)
        else:
            audio_data = fragment.get_audio('waveform')
            sr = fragment.get_buffer('waveform').sampling_rate
            midi_data = self.get_midi(audio_path=audio_path, audio=audio_data, sr=sr)
        if write:
            fragment.put_buffer(key=self.feature_name, b=pickle.dumps(midi_data), shape=None, unpickler=lambda x: pickle.loads(x))
        return midi_data

    def __call__(self, x):
        return self.get_midi(audio=x)
