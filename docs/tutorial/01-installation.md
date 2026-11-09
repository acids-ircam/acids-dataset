# 1. Installation

Install `acids-dataset` with pip :

```bash
pip install acids-dataset
```

`acids-dataset` requires Python 3.10 or above. Its scientific dependencies are installed with pip, with minimum versions `numpy>=1.23.5`, `scipy>=1.8`, `scikit-learn>=1.0.2` and `librosa>=0.10.2` (tested in CI, see [Development](../guide/development.md)).

Optional dependencies :

```bash
pip install acids-dataset[ui]       # dataset browser (gradio)
pip install torchcrepe pesto-pitch  # additional pitch estimators (crepe, pesto)
```

## The tutorial dataset

We generate a small dataset of three "instruments", with files named `{instrument}_{index}.wav` :

- `piano` : harmonic tones of different pitches,
- `bass` : low harmonic tones,
- `drums` : bursts of noise at a regular tempo.

```python
import math
from pathlib import Path
import torch, torchaudio

SR = 22050
root = Path("tutorial_dataset")
root.mkdir(exist_ok=True)
t = torch.arange(3 * SR) / SR

def tone(f0, n_partials=6, amp=0.3):
    return sum(amp / k * torch.sin(2 * math.pi * k * f0 * t) for k in range(1, n_partials + 1)) * torch.exp(-t)

def beat(bpm, amp=0.5):
    x = torch.zeros_like(t)
    for k in range(int(3 * bpm / 60)):
        start = int(k * 60 / bpm * SR)
        burst = torch.randn(int(0.05 * SR)) * torch.exp(-torch.arange(int(0.05 * SR)) / 200)
        x[start:start + len(burst)] += amp * burst[:len(x) - start]
    return x

for i, f0 in enumerate([262., 330., 392., 440., 523., 587.]):
    torchaudio.save(str(root / f"piano_{i}.wav"), tone(f0)[None], sample_rate=SR)
for i, f0 in enumerate([55., 65., 82.]):
    torchaudio.save(str(root / f"bass_{i}.wav"), tone(f0, amp=0.5)[None], sample_rate=SR)
for i, bpm in enumerate([90, 120]):
    torchaudio.save(str(root / f"drums_{i}.wav"), beat(bpm)[None], sample_rate=SR)

print(sorted(p.name for p in root.iterdir()))
```

We now have 11 files of 3 seconds. Next : [loading them as a dataset](02-first-dataset.md).
