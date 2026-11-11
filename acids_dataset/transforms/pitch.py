import numpy as np
from typing import Dict
import torchaudio
import torch
import pathlib
import os
import numpy as np
from .base import Transform
from .basic_pitch_torch.model import BasicPitchTorch
from .basic_pitch_torch.inference import predict


# key is model name, value is record path ; tuple with mirrors
_BASICPITCH_MODELS = {
    'icassp2022': ['basic_pitch_torch/assets/basic_pitch_pytorch_icassp_2022.pth', tuple()]
}


class BasicPitch(Transform):
    allow_random = False
    available_models = _BASICPITCH_MODELS
    def __init__(self, 
                 device="cpu", 
                 model: str = 'icassp2022', 
                 **kwargs
                ) -> None:
        kwargs['name'] = kwargs.get('name', "basic_pitch_"+model)
        super().__init__(**kwargs)
        self.pt_model = BasicPitchTorch()
        self.device = device
        self.load_model(model)

    def _try_download(self, model_path, mirrors):
        #TODO
        return False

    def load_model(self, model):
        if model not in self.available_models:
            raise ValueError('model %s not known. Available models : %s'%(self.available_models.keys()))
        file_path = pathlib.Path(__file__).parent.resolve()
        model_path, mirrors = self.available_models[model]
        model_path = os.path.join(file_path, model_path)
        if not os.path.exists(model_path):
            out = self._try_download(model_path, mirrors)
            if not out: raise RuntimeError('Could not download model %s. Please re-try, or select another model'%model)
        self.pt_model.load_state_dict(torch.load(model_path))
        self.pt_model.eval()
        self.pt_model.to(self.device)

    @torch.no_grad
    def __call__(self, waveform, sr = None, **kwargs):
        if type(waveform) != torch.Tensor:
            waveform = torch.from_numpy(waveform).to(device=self.device, dtype=torch.float32)
        sr = sr or self.sr
        if self.sr != 22050:
            waveform = torchaudio.functional.resample(waveform=waveform,
                                                      orig_freq=self.sr,
                                                      new_freq=22050)

        #print(waveform)
        if len(waveform.shape) > 1 and waveform.shape[0] > 1:
            results = []
            for wave in waveform:
                _, midi_data, _ = predict(model=self.pt_model,
                                          audio=wave.squeeze().cpu(),
                                          device=self.device)
                results.append(midi_data)
            return results
        else:
            _, midi_data, _ = predict(model=self.pt_model,
                                      audio=waveform.squeeze().cpu().float(),
                                      device=self.device)
            return midi_data
