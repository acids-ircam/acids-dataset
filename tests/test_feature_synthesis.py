"""Features are checked against the parameters of generated datasets (pitch and frequency of sines, gain, file names,
tempo of click tracks, stereo width), through pre-processed and online datasets."""
import math
import numpy as np
import pytest
import torch
import torchaudio
import librosa

from acids_dataset import preprocess_dataset
from acids_dataset.datasets import AudioDataset
from acids_dataset.features import F0, Pitch, Mel, CQT, Loudness, Width, BeatTrack

SR = 22050
NOTES = {"A3": 220., "C4": 261.63, "E4": 329.63, "A4": 440.}
GAINS = [-6., -12., -18., -24.]
WIDTHS = {"identical": 0., "opposite": 1., "uncorrelated": 0.5}
TEMPOS = [90, 120]
MODES = ["preprocessed", "online"]


def write(path, x, sr=SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), torch.as_tensor(x, dtype=torch.float32).reshape(-1, np.shape(x)[-1]), sample_rate=sr)


def sine(f, duration=2., amp=0.5, sr=SR):
    t = np.arange(int(duration * sr)) / sr
    return amp * np.sin(2 * np.pi * f * t)


def click_track(bpm, duration=8., sr=SR):
    t = np.arange(int(duration * sr)) / sr
    x = np.zeros_like(t)
    n = int(0.03 * sr)
    for k in range(int(duration * bpm / 60)):
        s = int(k * 60 / bpm * sr)
        f = 1500 if k % 4 == 0 else 1000
        x[s:s + n] += (1. if k % 4 == 0 else 0.6) * np.sin(2 * np.pi * f * t[:n]) * np.exp(-t[:n] * 80)
    return x[:len(t)]


def build(root, features_fn, mode, chunk_length, meta_regexp=None, channels=None):
    """builds a pre-processed or an online dataset from root, with fresh feature objects."""
    if mode == "preprocessed":
        out = root.parent / (root.name + "_preprocessed")
        preprocess_dataset(root, out=out, chunk_length=chunk_length, sample_rate=SR, channels=channels, features=features_fn(),
                           meta_regexp=meta_regexp or [], force=True)
        return AudioDataset(out)
    return AudioDataset(root, sr=SR, channels=channels, chunk_length=chunk_length, batch_by="dataset", features=features_fn(),
                        meta_regexp=meta_regexp or [], write_index=False)


def items(dataset):
    """yields (audio_path, fragment) for each item of a dataset."""
    for i in range(len(dataset)):
        fragment = dataset._fragment_from_index(i)
        yield fragment.get_metadata()['audio_path'], fragment


def to_numpy(x):
    return x.numpy() if torch.is_tensor(x) else np.asarray(x)


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    for note, f in NOTES.items():
        for take in range(2):
            write(root / "tonal" / f"{note}_{take}.wav", sine(f))
    for g in GAINS:
        write(root / "gains" / f"gain_{int(-g)}.wav", sine(997., amp=10 ** (g / 20)))
    rng = np.random.default_rng(0)
    a, b = 0.3 * rng.standard_normal(2 * SR), 0.3 * rng.standard_normal(2 * SR)
    write(root / "spatial" / "identical.wav", np.stack([a, a]))
    write(root / "spatial" / "opposite.wav", np.stack([a, -a]))
    write(root / "spatial" / "uncorrelated.wav", np.stack([a, b]))
    for bpm in TEMPOS:
        write(root / "beats" / f"click_{bpm}.wav", click_track(bpm))
    return root


def note_of(audio_path):
    return audio_path.split("/")[-1].split("_")[0]


# pitch and frequency

@pytest.mark.parametrize("mode", MODES)
def test_pitch_features(synthetic, mode):
    features = lambda: [F0(mode="yin", sr=SR, frame_length=2048, name="f0"), Pitch(mode="yin", sr=SR, frame_length=2048, name="pitch")]
    dataset = build(synthetic / "tonal", features, mode, chunk_length=SR)
    note_idx = {note: Pitch.idx_hash()[note[:-1]] for note in NOTES}
    n_items = 0
    for audio_path, fragment in items(dataset):
        note = note_of(audio_path)
        f0 = to_numpy(fragment.get("f0")).reshape(-1)
        f0 = f0[f0 > 0]
        assert abs(np.median(f0) - NOTES[note]) < 0.01 * NOTES[note], (audio_path, np.median(f0))
        pitch = to_numpy(fragment.get("pitch")).reshape(-1)
        assert np.bincount(pitch[pitch >= 0].astype(int)).argmax() == note_idx[note], (audio_path, pitch)
        n_items += 1
    assert n_items == len(dataset) > 0
    if mode == "preprocessed":
        # pitch classes are hashed : each file is indexed by its note name
        pitch_hash = dataset.feature_hash["pitch"]
        for note in NOTES:
            keys = {dataset._key_to_str(k) for k in pitch_hash[Pitch.note_hash()[note_idx[note]]]}
            file_keys = {k for path, ks in dataset.feature_hash["original_path"].items() if note_of(path) == note for k in map(dataset._key_to_str, ks)}
            assert file_keys <= keys


def test_pyin(synthetic):
    dataset = build(synthetic / "tonal", lambda: [F0(mode="pyin", sr=SR, frame_length=2048, name="f0")], "online", chunk_length=SR)
    for audio_path, fragment in list(items(dataset))[::4]:
        f0, voiced = to_numpy(fragment.get("f0")).reshape(-1, 2, to_numpy(fragment.get("f0")).shape[-1])[0]
        assert voiced.mean() > 0.9
        assert abs(np.median(f0[voiced > 0.5]) - NOTES[note_of(audio_path)]) < 0.01 * NOTES[note_of(audio_path)]


@pytest.mark.parametrize("mode", MODES)
def test_spectral_features(synthetic, mode):
    features = lambda: [Mel(sr=SR, n_fft=2048, name="mel"), CQT(sr=SR, name="cqt")]
    dataset = build(synthetic / "tonal", features, mode, chunk_length=SR)
    fbanks = torchaudio.functional.melscale_fbanks(1025, 0., SR / 2, 128, SR)
    for audio_path, fragment in items(dataset):
        f = NOTES[note_of(audio_path)]
        mel = to_numpy(fragment.get("mel"))
        mel_peak = int(mel.reshape(-1, 128, mel.shape[-1]).mean((0, 2)).argmax())
        assert abs(mel_peak - int(fbanks[int(round(f / SR * 2048))].argmax())) <= 1
        cqt = to_numpy(fragment.get("cqt"))
        n = cqt.shape[0] // 2
        magnitude = np.sqrt(cqt[:n] ** 2 + cqt[n:] ** 2)
        assert magnitude.mean((0, 2)).argmax() == round(12 * math.log2(f / librosa.note_to_hz("C1")))


# loudness

@pytest.mark.parametrize("mode", MODES)
def test_loudness(synthetic, mode):
    dataset = build(synthetic / "gains", lambda: [Loudness(sr=SR, keep_channels=False)], mode, chunk_length=SR)
    for audio_path, fragment in items(dataset):
        gain = -float(audio_path.split("_")[-1].split(".")[0])
        # a sine of amplitude A at 997 Hz has a loudness of 20 log10(A) - 3.01 LUFS
        assert abs(float(to_numpy(fragment.get("loudness"))) - (gain - 3.01)) < 0.2, audio_path


# metadata from paths

@pytest.mark.parametrize("mode", MODES)
def test_regexp(synthetic, mode):
    dataset = build(synthetic / "tonal", lambda: [], mode, chunk_length=SR, meta_regexp=["**/{{note}}_{{take}}.wav"])
    for audio_path, fragment in items(dataset):
        note, take = audio_path.split("/")[-1][:-4].split("_")
        if mode == "online":
            metadata = fragment.get_metadata()
            assert metadata["note"] == note and metadata["take"] == take
    hashed = dataset.feature_hash
    assert set(hashed["note"].keys()) == set(NOTES.keys()) and set(hashed["take"].keys()) == {"0", "1"}
    if mode == "online":
        assert all(len(v) == 2 for v in hashed["note"].values())
    else:
        files_per_note = {note: {p for p, ks in hashed["original_path"].items() if note_of(p) == note} for note in NOTES}
        for note, keys in hashed["note"].items():
            keys = set(map(dataset._key_to_str, keys))
            assert keys == {dataset._key_to_str(k) for p in files_per_note[note] for k in hashed["original_path"][p]}


# spatial

@pytest.mark.parametrize("mode", MODES)
def test_width(synthetic, mode):
    dataset = build(synthetic / "spatial", lambda: [Width(sr=SR, frame_length=2048)], mode, chunk_length=SR)
    for audio_path, fragment in items(dataset):
        kind = audio_path.split("/")[-1][:-4]
        width = to_numpy(fragment.get("width"))
        assert abs(width.mean() - WIDTHS[kind]) < 0.05, (kind, width.mean())


# beat tracking

@pytest.mark.parametrize("mode", MODES)
def test_beat_tracking(synthetic, mode):
    dataset = build(synthetic / "beats", lambda: [BeatTrack(sr=SR, downsample=256, name="beat")], mode, chunk_length=4 * SR)
    for audio_path, fragment in items(dataset):
        bpm = int(audio_path.split("_")[-1][:-4])
        beat_clock, downbeat_clock = to_numpy(fragment.get("beat")).reshape(2, -1)
        frame_rate = beat_clock.shape[-1] / 4.
        resets = np.where(np.diff(beat_clock) < -0.5)[0] + 1
        assert len(resets) >= 3, audio_path
        assert abs(np.median(np.diff(resets)) / frame_rate - 60 / bpm) < 0.05 * 60 / bpm, (audio_path, np.diff(resets))
        # downbeats are beats
        downbeats = np.where(np.diff(downbeat_clock) < -0.5)[0] + 1
        assert all(np.min(np.abs(resets - d)) <= 2 for d in downbeats)


def test_pesto(synthetic):
    from acids_dataset.features.pitch import _AD_F0_METHODS
    if "pesto" not in _AD_F0_METHODS:
        pytest.skip("pesto not installed")
    dataset = build(synthetic / "tonal", lambda: [F0(mode="pesto", sr=SR, frame_length=2048, hop_length=512, name="f0")], "online", chunk_length=SR)
    for audio_path, fragment in items(dataset):
        out = to_numpy(fragment.get("f0"))
        assert out.shape[-2:] == (2, (SR - 2048) // 512 + 1)
        f0, confidence = out.reshape(-1, 2, out.shape[-1])[0]
        assert abs(np.median(f0[confidence > 0.5]) - NOTES[note_of(audio_path)]) < 0.01 * NOTES[note_of(audio_path)]
