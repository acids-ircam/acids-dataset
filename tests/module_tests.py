from pathlib import Path
import os
import torch, torch.nn as nn
import gin.torch

scripted_paths = Path(__file__).parent / "scripted"

def check_scripted_version(module):
    if not hasattr(module, "scripted_name"):
        return
    os.makedirs(scripted_paths, exist_ok=True)
    if not(scripted_paths / module.scripted_name).exists():
        module_scripted = torch.jit.script(module())
        torch.jit.save(module_scripted, str(scripted_paths / module.scripted_name))

@gin.configurable(module="ad.test")
class ConvEmbedding(nn.Module):
    scripted_name = "conv_embedding.ts"
    def __init__(self):
        super().__init__()
        self.downsample = 16
        self.conv_module = nn.Conv1d(1, 16, 17, stride=self.downsample)

    @torch.jit.export 
    def get_embedding(self, x):
        return self.conv_module(x)

    def forward(self, x):
        return self.get_embedding(x)

check_scripted_version(ConvEmbedding)

