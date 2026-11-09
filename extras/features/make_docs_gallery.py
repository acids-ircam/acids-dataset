"""Generates the feature gallery of the documentation (docs/tutorial/03-features.md).

    python extras/features/make_docs_gallery.py

Figures are written in docs/tutorial/images/features. The test signal is synthetic : a stereo melody (C4 E4 G4 C5 B4 G4,
panned from left to right), kicks at 120 BPM, and a noise burst in the last second (delayed by 8 samples in the right channel).
"""
import math
import sys
from pathlib import Path
from typing import Any

# documents the code of this repository, even if another version of acids_dataset is installed
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from acids_dataset import features as F

SR = 22050
DURATION = 4.
OUT_DIR = REPO_ROOT / "docs" / "tutorial" / "images" / "features"
FRAME, HOP = 1024, 512
NOISE_DELAY = 8
NOTE_NAMES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
torch.manual_seed(0)


def make_signal():
    t = torch.arange(int(DURATION * SR)) / SR
    melody = torch.zeros_like(t)
    notes = [60, 64, 67, 72, 71, 67]
    note_length = 3. / len(notes)
    for i, midi in enumerate(notes):
        f0 = 440 * 2 ** ((midi - 69) / 12)
        tn = t - i * note_length
        env = ((tn >= 0) & (tn < note_length)).float() * torch.exp(-3 * tn.clamp(min=0)) * (1 - torch.exp(-200 * tn.clamp(min=0)))
        melody += env * sum(0.25 / k * torch.sin(2 * math.pi * f0 * k * tn) for k in range(1, 10))
    # kicks at 120 BPM
    kicks = torch.zeros_like(t)
    for beat in np.arange(0, DURATION, 0.5):
        tk = t - beat
        kicks += (tk >= 0).float() * torch.exp(-30 * tk.clamp(min=0)) * torch.sin(2 * math.pi * 60 * tk * torch.exp(-5 * tk.clamp(min=0)))
    # noise burst
    noise = 0.15 * torch.randn_like(t) * (t > 3.).float()
    # stereo : melody panned from left to right, kicks centered, noise delayed in the right channel
    pan = (t / DURATION).clamp(0, 1)
    left = melody * torch.cos(pan * math.pi / 2) + 0.5 * kicks + noise
    right = melody * torch.sin(pan * math.pi / 2) + 0.5 * kicks + torch.roll(noise, NOISE_DELAY)
    return t, torch.stack([left, right]) * 0.8


def times(n):
    return np.linspace(0, DURATION, n)


def spectrogram(ax, x, title="", ylim=None):
    spec = torch.stft(x.mean(0), FRAME, HOP // 2, window=torch.hann_window(FRAME), return_complex=True).abs()
    ax.imshow(20 * torch.log10(spec + 1e-6), origin="lower", aspect="auto", cmap="magma", vmin=-60, vmax=40,
              extent=[0, DURATION, 0, SR / 2])
    ax.set_ylim(*(ylim or (0, SR / 2)))
    ax.set_title(title, fontsize=10); ax.set_ylabel("Hz")


def curve(ax, values, title, color="C0", **kwargs):
    values = np.asarray(values).reshape(-1)
    ax.plot(times(len(values)), values, color=color, lw=1.2, **kwargs)
    ax.set_title(title, fontsize=10); ax.set_xlim(0, DURATION); ax.grid(alpha=0.3)


def image(ax, values, title, ylabel="", cmap="viridis", **kwargs):
    ax.imshow(np.asarray(values), origin="lower", aspect="auto", cmap=cmap, extent=[0, DURATION, 0, values.shape[0]], **kwargs)
    ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel)


def save(fig, name):
    for ax in fig.axes:
        if ax.get_subplotspec().is_last_row() and not ax.get_xlabel():
            ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=110, facecolor="white")
    plt.close(fig)
    print("written", OUT_DIR / name)


def kw(**kwargs) -> dict[str, Any]:
    return dict(frame_length=FRAME, hop_length=HOP, sr=SR, **kwargs)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t, x = make_signal()
    mono = x.mean(0, keepdim=True)

    # test signal
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
    for c, name in enumerate(["left", "right"]):
        axes[0].plot(t, x[c], lw=0.4, alpha=0.7, label=name)
    axes[0].legend(loc="upper left", fontsize=8); axes[0].set_title("test signal (stereo)", fontsize=10); axes[0].set_xlim(0, DURATION)
    spectrogram(axes[1], x, "spectrogram")
    save(fig, "signal.png")

    # spectral
    fig, axes = plt.subplots(4, 2, figsize=(12, 9), sharex=True)
    spectrogram(axes[0, 0], x, "SpectralCentroid / SpectralRolloff (Hz)", ylim=(0, 8000))
    for feat, style in [(F.SpectralCentroid(**kw()), "c"), (F.SpectralRolloff(**kw()), "w--")]:
        v = feat(mono)
        axes[0, 0].plot(times(len(v)), v, style, lw=1.2, label=type(feat).__name__)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    curve(axes[0, 1], F.SpectralSpread(**kw())(mono), "SpectralSpread (Hz)")
    curve(axes[1, 0], F.SpectralSkewness(**kw())(mono), "SpectralSkewness")
    curve(axes[1, 1], F.SpectralKurtosis(**kw())(mono), "SpectralKurtosis")
    curve(axes[2, 0], F.SpectralFlatness(**kw())(mono), "SpectralFlatness")
    curve(axes[2, 1], F.SpectralCrest(**kw())(mono), "SpectralCrest")
    curve(axes[3, 0], F.SpectralFlux(**kw())(mono), "SpectralFlux")
    image(axes[3, 1], F.SpectralContrast(**kw())(mono), "SpectralContrast (dB)", "octave band")
    save(fig, "spectral.png")

    # temporal & loudness
    fig, axes = plt.subplots(2, 2, figsize=(12, 4.5), sharex=True)
    curve(axes[0, 0], F.RMS(**kw())(mono), "RMS (dB)")
    curve(axes[0, 1], F.Loudness(frame_length=FRAME, sr=SR, keep_channels=False)(mono), "Loudness (LUFS)")
    curve(axes[1, 0], F.ZeroCrossingRate(**kw(unit="hz"))(mono), "ZeroCrossingRate (Hz)")
    curve(axes[1, 1], F.OnsetStrength(**kw())(mono), "OnsetStrength")
    tempo, rate = F.Tempo(sr=SR)(mono), F.OnsetRate(sr=SR)(mono)
    axes[1, 1].set_title("OnsetStrength  —  Tempo : %.1f BPM, OnsetRate : %.1f onsets / s" % (tempo, rate), fontsize=10)
    save(fig, "temporal.png")

    # pitch & harmony
    fig, axes = plt.subplots(3, 2, figsize=(12, 7), sharex=True)
    spectrogram(axes[0, 0], x, "F0 (yin, Hz)", ylim=(0, 2000))
    f0 = F.F0(mode="yin", sr=SR, fmin=100, fmax=1000)(mono)[0]
    f0 = np.where(np.asarray(f0) > 0, np.asarray(f0), np.nan)
    axes[0, 0].plot(times(len(f0)), f0, "c", lw=1.5)
    curve(axes[0, 1], F.Pitch(mode="yin", sr=SR, fmin=100, fmax=1000)(mono)[0], "Pitch (pitch class)", drawstyle="steps-mid")
    axes[0, 1].set_yticks(np.arange(12), NOTE_NAMES, fontsize=7)
    image(axes[1, 0], F.Chroma(**kw())(mono), "Chroma", "pitch class")
    axes[1, 0].set_yticks(np.arange(12) + 0.5, ["C", "", "D", "", "E", "F", "", "G", "", "A", "", "B"], fontsize=7)
    curve(axes[1, 1], F.Harmonicity(**kw())(mono), "Harmonicity")
    curve(axes[2, 0], F.Inharmonicity(**kw())(mono), "Inharmonicity")
    axes[2, 1].axis("off")
    save(fig, "pitch_harmony.png")

    # representations
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    mel = F.Mel(sr=SR, n_fft=FRAME, hop_length=HOP // 2, n_mels=128)(mono)[0]
    image(axes[0, 0], 10 * torch.log10(mel + 1e-8), "Mel (dB)", "mel bin", cmap="magma", vmin=-60)
    cqt = F.CQT(sr=SR, hop_length=HOP // 2)(mono)
    image(axes[0, 1], 20 * np.log10(np.hypot(cqt[0], cqt[1]) + 1e-6), "CQT (magnitude, dB)", "bin", cmap="magma", vmin=-60)
    gammatone = F.Gammatone(**kw(n_bands=64, db=True))(mono)
    image(axes[1, 0], gammatone, "Gammatone (dB, 64 bands on the ERB scale)", "band", cmap="magma", vmin=-40)
    dct = F.DCT(**kw(n_coeffs=256))(mono)
    image(axes[1, 1], 20 * torch.log10(dct.abs() + 1e-6), "DCT (first 256 coefficients, dB)", "coefficient", cmap="magma", vmin=-40)
    cepstrum = F.Cepstrum(**kw(n_coeffs=200))(mono)
    axes[2, 0].imshow(np.asarray(cepstrum[20:]), origin="lower", aspect="auto", cmap="viridis", vmin=0, extent=[0, DURATION, 20, 200])
    axes[2, 0].set_title("Cepstrum (quefrencies 20 to 200 samples)", fontsize=10); axes[2, 0].set_ylabel("quefrency (samples)")
    axes[2, 0].set_xlabel("time (s)")
    # LPC envelope on the first note
    start = int(0.1 * SR)
    segment = mono[:, start:start + FRAME]
    spectrum = 10 * torch.log10(torch.fft.rfft(segment[0] * torch.hann_window(FRAME)).abs().square() + 1e-10)
    envelope = F.LPC(**kw(order=24, mode="envelope"))(segment)[:, 0]
    freqs = np.fft.rfftfreq(FRAME, 1 / SR)
    axes[2, 1].plot(freqs, spectrum, lw=0.6, color="0.6", label="power spectrum")
    axes[2, 1].plot(freqs, envelope, lw=1.5, color="C3", label="LPC envelope (order 24)")
    axes[2, 1].set_xlim(0, 5000); axes[2, 1].set_ylim(float(spectrum.max()) - 90, float(spectrum.max()) + 10)
    axes[2, 1].set_title("LPC envelope (first note)", fontsize=10); axes[2, 1].set_xlabel("frequency (Hz)"); axes[2, 1].set_ylabel("dB")
    axes[2, 1].legend(loc="upper right", fontsize=8); axes[2, 1].grid(alpha=0.3)
    save(fig, "representations.png")

    # Fourier transforms and filter banks
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    stft = F.STFT(frame_length=FRAME, hop_length=HOP // 2, sr=SR, mode=["db", "phase", "instantaneous_frequency", "instantaneous_phase"])
    db, phase, frequency, inst_phase = stft(mono)
    image(axes[0, 0], db, "STFT (mode db)", "bin", cmap="magma", vmin=-40)
    image(axes[0, 1], phase, "STFT (mode phase)", "bin", cmap="twilight")
    # instantaneous frequencies, where the spectrum is loud enough
    loud = db > db.max() - 50
    times_stft = times(db.shape[-1])[None].repeat(db.shape[0], 0)
    axes[1, 0].scatter(times_stft[loud.numpy()], frequency[loud].numpy(), s=0.3, c=db[loud].numpy(), cmap="magma", vmin=float(db.max()) - 50)
    axes[1, 0].set_ylim(0, 3000); axes[1, 0].set_xlim(0, DURATION)
    axes[1, 0].set_title("STFT (mode instantaneous_frequency, bins within 50 dB of the max)", fontsize=10); axes[1, 0].set_ylabel("Hz")
    axes[1, 0].set_xlabel("time (s)")
    image(axes[1, 1], inst_phase[:200], "STFT (mode instantaneous_phase, first 200 bins)", "bin", cmap="twilight")
    axes[1, 1].set_xlabel("time (s)")
    fft = F.FFT(sr=SR, mode="power_density", window="hann")
    density = fft(mono)
    axes[2, 0].semilogx(fft.frequencies(mono.shape[-1])[1:], 10 * torch.log10(density[1:] + 1e-12), lw=0.5)
    axes[2, 0].set_xlim(20, SR / 2); axes[2, 0].grid(alpha=0.3)
    axes[2, 0].set_title("FFT (mode power_density, whole signal)", fontsize=10); axes[2, 0].set_xlabel("frequency (Hz)"); axes[2, 0].set_ylabel("dB / Hz")
    bands = F.PQMF(n_bands=8, sr=SR)(mono)
    image(axes[2, 1], 20 * torch.log10(bands.abs() + 1e-4), "PQMF (8 bands, magnitude in dB)", "band", cmap="magma", vmin=-60)
    save(fig, "fourier.png")

    # spatial
    fig, axes = plt.subplots(3, 2, figsize=(12, 7), sharex=True)
    curve(axes[0, 0], F.Pan(**kw())(x), "Pan (-1 : left, 1 : right)")
    curve(axes[0, 1], F.Width(**kw())(x), "Width")
    curve(axes[1, 0], F.InterChannelLevelDifference(**kw())(x), "InterChannelLevelDifference (dB)")
    curve(axes[1, 1], F.InterChannelCoherence(**kw())(x), "InterChannelCoherence")
    curve(axes[2, 0], F.InterChannelTimeDifference(**kw())(x), "InterChannelTimeDifference (samples)")
    gcc_feature = F.GeneralizedCrossCorrelation(**kw(mode="correlation"))
    gcc = gcc_feature(x)[0]
    lags = np.asarray(gcc_feature.lags)
    axes[2, 1].imshow(np.asarray(gcc), origin="lower", aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                      extent=[0, DURATION, lags[0] - 0.5, lags[-1] + 0.5])
    axes[2, 1].set_title("GeneralizedCrossCorrelation (PHAT)", fontsize=10); axes[2, 1].set_ylabel("lag (samples)")
    save(fig, "spatial.png")


if __name__ == "__main__":
    main()
