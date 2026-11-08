"""Health reports of datasets : clipping, DC offset, silence and level of each file (pre-processed or online datasets),
with flags for problematic files."""
import math
from typing import Any, Dict, List, Optional

import torch

from .features.framed import _EPS, _frames

DEFAULT_THRESHOLDS = {
    "clipping_ratio": 1e-4,   # ratio of clipped samples above which files are flagged
    "dc_offset": 0.01,        # absolute DC offset above which files are flagged
    "silence_ratio": 0.5,     # ratio of silent frames above which files are flagged as mostly silent
    "peak_db": -40.,          # peak level below which files are flagged as very quiet
}


def audio_health(audio, sr: int, clipping_threshold: float = 0.999, silence_db: float = -60., frame_length: int = 2048) -> Dict[str, float]:
    """health descriptors of audio (channels, samples) : duration, peak and RMS levels (dB), clipping ratio, maximum
    absolute DC offset over channels, and ratio of silent frames."""
    x = torch.as_tensor(audio).float()
    if x.ndim == 1: x = x[None]
    if x.shape[-1] == 0:
        return {'duration': 0., 'peak_db': -math.inf, 'rms_db': -math.inf, 'clipping_ratio': 0., 'dc_offset': 0., 'silence_ratio': 1.}
    peak = float(x.abs().max())
    rms = float(torch.sqrt(x.square().mean()))
    frames_rms = torch.sqrt(_frames(x.mean(0, keepdim=True), frame_length, frame_length).square().mean(-1))
    return {
        'duration': x.shape[-1] / sr,
        'peak_db': 20 * math.log10(peak) if peak > _EPS else -math.inf,
        'rms_db': 20 * math.log10(rms) if rms > _EPS else -math.inf,
        'clipping_ratio': float((x.abs() >= clipping_threshold).float().mean()),
        'dc_offset': float(x.mean(-1).abs().max()),
        'silence_ratio': float((20 * torch.log10(frames_rms + _EPS) < silence_db).float().mean()),
    }


def health_flags(health: Dict[str, float], thresholds: Dict[str, float] | None = None) -> List[str]:
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    flags = []
    if health['silence_ratio'] >= 1.:
        flags.append("silent")
    elif health['silence_ratio'] > t['silence_ratio']:
        flags.append("mostly silent")
    if health['clipping_ratio'] > t['clipping_ratio']:
        flags.append("clipping")
    if health['dc_offset'] > t['dc_offset']:
        flags.append("dc offset")
    if health['peak_db'] < t['peak_db'] and health['silence_ratio'] < 1.:
        flags.append("very quiet")
    return flags


def dataset_health(dataset, max_files: int | None = None, thresholds: Dict[str, float] | None = None, progress: bool = False, **kwargs) -> Dict[str, Any]:
    """health report of the files of a dataset (audio is loaded with dataset.load_audio, at the sampling rate of the
    dataset ; audio of pre-processed datasets is rebuilt from stored chunks). Returns {"files": [...], "summary": {...}},
    where files give the descriptors of audio_health and their flags, and summary the number of files per flag."""
    files = dataset.files
    indices = list(range(len(files)))[:max_files] if max_files is not None else list(range(len(files)))
    iterator = indices
    if progress:
        import tqdm
        iterator = tqdm.tqdm(indices, desc="dataset health")
    sr = int(files[0]['sr']) if files else None
    reports = []
    for i in iterator:
        audio = dataset.load_audio(i)
        file_sr = getattr(dataset, "sr", None) or files[i]['sr']
        health = audio_health(audio, int(file_sr), **kwargs)
        reports.append({'index': i, 'audio_path': files[i]['audio_path'], **health, 'flags': health_flags(health, thresholds)})
    counts = {}
    for r in reports:
        for f in r['flags']:
            counts[f] = counts.get(f, 0) + 1
    summary = {'n_files': len(reports), 'n_flagged': sum(1 for r in reports if r['flags']), 'flags': counts,
               'total_duration': sum(r['duration'] for r in reports), 'thresholds': {**DEFAULT_THRESHOLDS, **(thresholds or {})}}
    return {'files': reports, 'summary': summary}


__all__ = ['audio_health', 'health_flags', 'dataset_health', 'DEFAULT_THRESHOLDS']
