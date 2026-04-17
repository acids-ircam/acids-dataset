import numpy as np
import tempfile
from pathlib import Path
import torch, torchaudio, gin.torch
from .base import AcidsDatasetFeature, FileHash

Audio2Beats = None

@gin.configurable(module="features")
class BeatTrack(AcidsDatasetFeature):
    def __init__(self, 
                 sr=None,
                 device="cpu", 
                 checkpoint_path="final0", 
                 dbn=False, 
                 downsample: int | None = None,
                 float16=False, 
                 **kwargs) -> None:
        self._downsample = downsample
        super().__init__(device=device, **kwargs)
        self.sr = sr
        self._tmp_folder = Path(tempfile.mkdtemp()).resolve()
        self._file_buffer = FileHash()
        global Audio2Beats
        if Audio2Beats is None: 
            from .beat_this.inference import Audio2Beats
        self.audio2beats = Audio2Beats(checkpoint_path=checkpoint_path,
                                       dbn=dbn,
                                       float16=float16,
                                       device=device)

    @property
    def default_feature_name(self): 
        return f"beat_{self._downsample}"

    def get_beat_signal(self, b, len_wave, len_z, sr=24000, zero_value=0):
        if len(b) < 2:
            #print("empty beat")
            return zero_value * np.ones(len_z)
        times = np.linspace(0, len_wave / sr, len_z)
        t_max = times[-1]
        i = 0
        while i < len(b) - 1 and b[i] < t_max:
            i += 1
        b = b[:i]
        minvalue = 0
        id_time_min = 0
        out = []

        if len(b) < 3:
            #print("empty beat")
            return np.zeros(len(times))
        for i in range(len(b)):
            time = b[i]
            time_prev = b[i - 1] if i > 0 else 0
            delt = time - times

            try:
                id_time_max = np.argmin(delt[delt > 0])
                time_interp = times[id_time_max]
                maxvalue = (time_interp - time_prev) / (time - time_prev)
            except:
                id_time_max = 1
                maxvalue = 1

            out.append(
                np.linspace(minvalue, maxvalue, 1 + id_time_max - id_time_min))

            if i < len(b) - 1:
                minvalue = (times[id_time_max + 1] - time) / (b[i + 1] - time)
                id_time_min = id_time_max + 1

        maxvalue = (times[len_z - 1] - time) / (time - time_prev)
        minvalue = (times[id_time_max] - time) / (time - time_prev)
        id_time_min = id_time_max + 1
        out.append(np.zeros(1 + len_z - id_time_min))

        out = np.concatenate(out)
        out = out[:len(times)]
        if len(out) < len(times):
            out = np.concatenate((out, np.zeros(abs(len(times) - len(out)))))
        return out

    def track_beat(self, audio: np.ndarray | None, audio_path: str | Path | None = None, z_length: int | None = None, sr: int | None = None):
        z_length = z_length or audio.shape[-1] // self._downsample
        sr = sr or self.sr
        if audio is not None:
            if isinstance(audio, torch.Tensor):
                waveform = audio.numpy()
            else:
                waveform = audio 
        elif audio_path is not None: 
            x, x_sr = torchaudio.load(audio_path)
            if sr != x_sr: x = torchaudio.functional.resample(x, x_sr, sr)
            x = x.numpy()
        else:
            raise ValueError('either audio or audio_path must be given.')
        if waveform.ndim < 3:
            beats, downbeats = self.audio2beats(waveform.transpose(), sr)
            beat_clock = self.get_beat_signal(beats,
                                            waveform.shape[-1],
                                            z_length,
                                            sr=sr,
                                            zero_value=0.)
            downbeat_clock = self.get_beat_signal(downbeats,
                                                waveform.shape[-1],
                                                z_length,
                                                sr=sr,
                                                zero_value=0.)        
        else:
            batch_shape = waveform.shape[:-2]
            waveform = np.reshape(waveform, (-1,)+ waveform.shape[-2:])
            beat_clock, downbeat_clock = [], []
            for w in waveform:
                b, d = self.audio2beats(w.transpose(), sr)
                bc = self.get_beat_signal(b,
                                        waveform.shape[-1],
                                        z_length,
                                        sr=sr,
                                        zero_value=0.)
                dc = self.get_beat_signal(d,
                                        waveform.shape[-1],
                                        z_length,
                                        sr=sr,
                                        zero_value=0.)
                beat_clock.append(bc)
                downbeat_clock.append(dc)
            beat_clock = np.stack(beat_clock)
            downbeat_clock = np.stack(downbeat_clock)
            beat_clock = np.reshape(beat_clock, batch_shape + (beat_clock.shape[-1],))
            downbeat_clock = np.reshape(downbeat_clock, batch_shape + (downbeat_clock.shape[-1],))
        return np.stack([beat_clock, downbeat_clock], axis=0)

    def _write_beat_tracking(self, path, track_data):
        beat_path = self._tmp_folder / f"{self._file_buffer.current_id:09d}.npz"
        np.savez_compressed(beat_path, beat=track_data)
        self._file_buffer[path] = beat_path
        
    def _extract_from_file(self, beat_path, sr=None, start=None, end=None, length=None):
        with np.load(beat_path) as data:
            if (start is not None) or (end is not None):
                assert sr is not None, "sr is required if _extract_from_file is provided a start or a end keyword."
                assert length is not None, "length is required if _extract_from_file is provided a start or a end keyword."
                start = None if start is None else int((start * data['beat'].shape[-1]) / length)
                end = None if end is None else int((end * data['beat'].shape[-1]) / length)
            clock = data["beat"][slice(start, end)]
        return clock

    def pre_chunk_hook(self, path, audio, sr, dataset_path):
        track_data = self.track_beat(audio_path=path, audio=audio, sr=sr)
        self._write_beat_tracking(path, track_data)

    def from_fragment(self, fragment, write: bool = True):
        audio_path = getattr(fragment, "audio_path", None)
        assert audio_path, f"{type(self)} requires an initialized audio_path metadata from fragment"
        if audio_path in self._file_buffer:
            sr = fragment.get_buffer('waveform').sampling_rate
            metadata = fragment.get_metadata()
            start_pos = float(metadata['start_pos']); length = float(metadata['file_length']); end_pos = start_pos + float(metadata['length'])
            beat_data = self._extract_from_file(self._file_buffer[audio_path], sr, start_pos, end_pos, length)
        else:
            audio_data = fragment.get_audio('waveform')
            sr = fragment.get_buffer('waveform').sampling_rate
            beat_data = self.track_beat(audio_path=audio_path, audio=audio_data, sr=sr)
        if write:
            fragment.put_array(self.feature_name, beat_data, sr=sr)
        return beat_data

    def __call__(self, x):
        out = self.track_beat(audio=x.cpu(), sr=self.sr)
        return torch.from_numpy(out).to(x)