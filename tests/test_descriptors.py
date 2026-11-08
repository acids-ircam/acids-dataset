"""Audio descriptors (spectral, temporal, onsets, harmonic, spatial, quality) checked on generated signals with known
properties, batch / item consistency, and dataset health reports."""
import math
import numpy as np
import pytest
import scipy.signal
import torch
import torchaudio

import acids_dataset.features as F
from acids_dataset import preprocess_dataset
from acids_dataset.datasets import AudioDataset
from acids_dataset.health import audio_health, health_flags, dataset_health

SR = 22050
N = 2 * SR
T = torch.arange(N) / SR


def sine(f, amp=0.5, n=N):
    return amp * torch.sin(2 * math.pi * f * torch.arange(n) / SR)[None]


def noise(seed=0, amp=0.3, channels=1):
    return amp * torch.randn(channels, N, generator=torch.Generator().manual_seed(seed))


def harmonic(f0, n_partials=8, stretch=0.):
    return sum((0.5 / k) * torch.sin(2 * math.pi * k * f0 * math.sqrt(1 + stretch * k * k) * T) for k in range(1, n_partials + 1))[None]


def clicks(bpm, duration=8.):
    t = torch.arange(int(duration * SR)) / SR
    x = torch.zeros_like(t)
    n = int(0.03 * SR)
    for k in range(int(duration * bpm / 60)):
        s = int(k * 60 / bpm * SR)
        x[s:s + n] += torch.sin(2 * math.pi * 1000 * t[:n]) * torch.exp(-t[:n] * 80)
    return x[None]


def mean(x):
    return float(torch.as_tensor(x).float().mean())


# spectral

@pytest.mark.parametrize("f", [250., 1000., 4000.])
def test_centroid(f):
    assert abs(mean(F.SpectralCentroid(sr=SR)(sine(f))) - f) < 0.01 * f
    assert abs(mean(F.SpectralCentroid(sr=SR, unit="normalized")(sine(f))) - f / (SR / 2)) < 0.01
    assert abs(mean(F.SpectralCentroid(sr=SR, unit="midi")(sine(440.))) - 69) < 0.05
    assert abs(mean(F.SpectralCentroid(sr=SR)(noise())) - SR / 4) < 0.03 * SR / 4
    with pytest.raises(AssertionError):
        F.SpectralSpread(unit="midi")


def test_spectral_shape():
    lowpassed = torch.from_numpy(scipy.signal.lfilter(*scipy.signal.butter(8, 2000 / (SR / 2)), noise().numpy())).float()
    assert abs(mean(F.SpectralRolloff(sr=SR)(lowpassed)) - 0.85 * 2000) < 100
    assert mean(F.SpectralRolloff(sr=SR, percentile=0.5)(lowpassed)) < mean(F.SpectralRolloff(sr=SR)(lowpassed))
    # flatness : ~0 for tones, e^-gamma (0.56) for the periodogram of white noise
    assert mean(F.SpectralFlatness(sr=SR)(sine(1000.))) < 0.01
    assert abs(mean(F.SpectralFlatness(sr=SR)(noise())) - math.exp(-0.5772)) < 0.05
    assert mean(F.SpectralFlatness(sr=SR, db=True)(noise())) < 0
    assert mean(F.SpectralSpread(sr=SR)(sine(1000.))) < 20 and mean(F.SpectralSpread(sr=SR)(noise())) > 2500
    assert abs(mean(F.SpectralSkewness(sr=SR)(noise()))) < 0.1
    # kurtosis of a uniform distribution : 1.8
    assert abs(mean(F.SpectralKurtosis(sr=SR)(noise())) - 1.8) < 0.1
    assert mean(F.SpectralCrest(sr=SR)(sine(1000.))) > 50 * mean(F.SpectralCrest(sr=SR)(noise()))
    contrast = F.SpectralContrast(sr=SR, n_bands=6)
    assert contrast(noise()).shape == (6, N // 2048) and mean(contrast(harmonic(220.))) > mean(contrast(noise()))


def test_flux():
    jump = torch.cat([sine(500.)[:, :SR], sine(2000.)[:, SR:]], -1)
    flux = F.SpectralFlux(sr=SR)(jump)
    assert abs(int(flux.argmax()) - SR / 2048) <= 1.5
    assert float(F.SpectralFlux(sr=SR)(sine(500.))[1:].max()) < 1e-3 and float(flux[0]) == 0


def test_silence():
    silence = torch.zeros(1, N)
    for feature in [F.SpectralCentroid, F.SpectralSpread, F.SpectralSkewness, F.SpectralKurtosis, F.SpectralRolloff, F.SpectralFlatness,
                    F.SpectralCrest, F.SpectralFlux, F.ZeroCrossingRate, F.Harmonicity, F.Inharmonicity, F.TemporalCentroid, F.AttackTime]:
        assert torch.all(feature(sr=SR)(silence) == 0), feature.__name__
    assert torch.all(F.RMS(sr=SR)(silence) == -100.)


# temporal

def test_temporal():
    assert abs(mean(F.RMS(sr=SR)(sine(1000.))) - 20 * math.log10(0.5 / math.sqrt(2))) < 0.01
    assert abs(mean(F.RMS(sr=SR, db=False)(sine(1000.))) - 0.5 / math.sqrt(2)) < 1e-3
    for f in [220., 1000.]:
        assert abs(mean(F.ZeroCrossingRate(sr=SR, unit="hz")(sine(f))) - f) < 0.01 * f
        assert abs(mean(F.ZeroCrossingRate(sr=SR)(sine(f))) - 2 * f / SR) < 1e-3
    assert abs(mean(F.CrestFactor(sr=SR)(sine(1000.))) - 10 * math.log10(2)) < 0.01


def test_envelope():
    rise = int(0.2 * SR)
    envelope = torch.cat([torch.linspace(0, 1, rise), torch.exp(-torch.arange(N - rise) / SR * 3)])
    x = envelope * torch.sin(2 * math.pi * 1000 * T)
    # 10% -> 90% of a linear rise of 0.2s : 0.16s
    assert abs(mean(F.AttackTime(sr=SR)(x[None])) - 0.16) < 0.02
    assert abs(mean(F.TemporalCentroid(sr=SR)(sine(1000.))) - 1.) < 0.02
    assert mean(F.TemporalCentroid(sr=SR)(x[None])) < 1.


# onsets

@pytest.mark.parametrize("bpm", [90, 120, 150])
def test_tempo(bpm):
    x = clicks(bpm)
    assert abs(mean(F.Tempo(sr=SR)(x)) - bpm) < 0.02 * bpm
    # the click at the beginning of the chunk has no previous frame
    n_clicks = int(8 * bpm / 60)
    assert abs(mean(F.OnsetRate(sr=SR)(x)) - (n_clicks - 1) / 8) <= 1 / 8 + 1e-6
    strength = F.OnsetStrength(sr=SR)(x)
    assert strength.shape[-1] == (x.shape[-1] - 1024) // 256 + 1


# harmonic

def test_chroma():
    for note, idx in [(261.63, 0), (440., 9), (329.63, 4)]:
        chroma = F.Chroma(sr=SR)(sine(note))
        assert chroma.shape[0] == 12 and int(chroma.mean(-1).argmax()) == idx
    assert torch.allclose(F.Chroma(sr=SR, norm="sum")(harmonic(220.)).sum(0), torch.ones(1), atol=1e-4)


def test_harmonicity():
    assert mean(F.Harmonicity(sr=SR)(sine(220.))) > 0.99 and mean(F.Harmonicity(sr=SR)(harmonic(110.))) > 0.95
    assert mean(F.Harmonicity(sr=SR)(noise())) < 0.2


def test_inharmonicity():
    for f0 in [110., 220., 440.]:
        assert mean(F.Inharmonicity(sr=SR)(harmonic(f0))) < 1e-3
    values = [mean(F.Inharmonicity(sr=SR)(harmonic(220., stretch=b))) for b in [0.001, 0.002, 0.005]]
    assert values == sorted(values) and values[0] > 3e-3


# spatial

def test_spatial_descriptors():
    a, b = noise(0)[0], noise(1)[0]
    quieter = torch.stack([a, 0.5 * a])
    assert abs(mean(F.InterChannelLevelDifference(sr=SR)(quieter)) + 20 * math.log10(2)) < 0.01
    assert F.InterChannelLevelDifference(sr=SR, n_bands=4)(quieter).shape == (1, 4, N // 2048)
    assert abs(mean(F.Pan(sr=SR)(quieter)) - (0.25 - 1) / 1.25) < 1e-3
    assert mean(F.Pan(sr=SR)(torch.stack([torch.zeros(N), a]))) == pytest.approx(1.)
    for delay in [3, 7, -5]:
        delayed = torch.stack([a, torch.roll(a, delay)])
        assert abs(mean(F.InterChannelTimeDifference(sr=SR)(delayed)) - delay) < 0.2
    assert abs(mean(F.InterChannelTimeDifference(sr=SR, unit="seconds")(torch.stack([a, torch.roll(a, 7)]))) - 7 / SR) < 1e-5
    assert mean(F.InterChannelCoherence(sr=SR)(torch.stack([a, torch.roll(a, 7)]))) > 0.95
    assert mean(F.InterChannelCoherence(sr=SR)(torch.stack([a, b]))) < 0.15
    # mono inputs give neutral values
    assert mean(F.Pan(sr=SR)(a[None])) == 0 and mean(F.InterChannelCoherence(sr=SR)(a[None])) == pytest.approx(1.)
    assert F.InterChannelTimeDifference(sr=SR)(torch.stack([a, a, b])).shape == (2, N // 2048)


# quality

def test_quality():
    clipped = sine(440., 2.).clamp(-1, 1)
    assert abs(mean(F.ClippingRatio(sr=SR)(clipped)) - 2 / 3) < 0.01 and mean(F.ClippingRatio(sr=SR)(sine(440.))) == 0
    stereo_clip = torch.cat([clipped, sine(440.)])
    assert F.ClippingRatio(sr=SR, keep_channels=True)(stereo_clip).shape == (2,) and abs(mean(F.ClippingRatio(sr=SR)(stereo_clip)) - 1 / 3) < 0.01
    assert F.DCOffset(sr=SR)(torch.cat([sine(440.) + 0.1, sine(440.) - 0.2])).tolist() == pytest.approx([0.1, -0.2], abs=1e-3)
    half_silent = torch.cat([sine(440.)[:, :SR], torch.zeros(1, SR)], -1)
    assert abs(mean(F.SilenceRatio(sr=SR)(half_silent)) - 0.5) < 0.05
    steps = torch.cat([sine(440., 0.5)[:, :SR], sine(440., 0.05)[:, SR:]], -1)
    assert abs(mean(F.DynamicRange(sr=SR)(steps)) - 20) < 0.5 and mean(F.DynamicRange(sr=SR)(torch.zeros(1, N))) == 0


# consistency

ALL_DESCRIPTORS = [F.SpectralCentroid, F.SpectralSpread, F.SpectralSkewness, F.SpectralKurtosis, F.SpectralRolloff, F.SpectralFlatness,
                   F.SpectralCrest, F.SpectralFlux, F.SpectralContrast, F.RMS, F.ZeroCrossingRate, F.AttackTime, F.TemporalCentroid,
                   F.CrestFactor, F.OnsetStrength, F.OnsetRate, F.Tempo, F.Chroma, F.Harmonicity, F.Inharmonicity,
                   F.InterChannelLevelDifference, F.InterChannelTimeDifference, F.InterChannelCoherence, F.GeneralizedCrossCorrelation, F.Pan, F.CrossSpectralMatrix, F.SpectroSpatialCovarianceVector,
                   F.ClippingRatio, F.DCOffset, F.DynamicRange, F.SilenceRatio]


@pytest.mark.parametrize("feature_class", ALL_DESCRIPTORS, ids=lambda c: c.__name__)
def test_batch_and_items(feature_class):
    feature = feature_class(sr=SR)
    x = torch.stack([noise(0, channels=2), torch.cat([harmonic(220.), harmonic(330.)]), torch.cat([clicks(120, 2.), clicks(120, 2.)])])
    batch = feature.batch_call(x)
    items = torch.stack([feature(x[i]) for i in range(len(x))])
    assert feature.supports_batch and batch.shape == items.shape and torch.allclose(batch, items, atol=1e-4, rtol=1e-4)
    # short inputs give one frame (or one value per chunk)
    short = feature(torch.randn(2, 100) * 0.1)
    assert short.numel() > 0 and torch.all(torch.isfinite(short))


@pytest.mark.parametrize("feature_class", [F.SpectralCentroid, F.RMS, F.Chroma, F.SpectralContrast, F.Tempo, F.ClippingRatio], ids=lambda c: c.__name__)
def test_keep_channels(feature_class):
    x = torch.cat([sine(440.), sine(880.)])
    mixed, channels = feature_class(sr=SR)(x), feature_class(sr=SR, keep_channels=True)(x)
    assert channels.shape == (2,) + mixed.shape


@pytest.mark.parametrize("mode", ["online", "preprocessed"])
def test_descriptors_in_datasets(tmp_path, mode):
    for f in [440., 880.]:
        torchaudio.save(str(tmp_path / f"{int(f)}.wav"), sine(f), sample_rate=SR)
    features = lambda: [F.SpectralCentroid(sr=SR), F.ZeroCrossingRate(sr=SR, unit="hz"), F.RMS(sr=SR)]
    if mode == "online":
        dataset = AudioDataset(tmp_path, sr=SR, chunk_length=SR, features=features(), write_index=False)
    else:
        preprocess_dataset(tmp_path, out=tmp_path.parent / (tmp_path.name + "_pre"), chunk_length=SR, sample_rate=SR, features=features(), force=True)
        dataset = AudioDataset(tmp_path.parent / (tmp_path.name + "_pre"))
    for i in range(len(dataset)):
        fragment = dataset._fragment_from_index(i)
        f = float(fragment.get_metadata()['audio_path'].split("/")[-1][:-4])
        assert abs(mean(fragment.get("spectral_centroid")) - f) < 0.01 * f
        assert abs(mean(fragment.get("zero_crossing_rate")) - f) < 0.01 * f


# health

def test_audio_health():
    report = audio_health(sine(440., 1.5).clamp(-1, 1), SR)
    assert report['clipping_ratio'] > 0.3 and "clipping" in health_flags(report)
    assert health_flags(audio_health(sine(440.) + 0.1, SR)) == ["dc offset"]
    assert health_flags(audio_health(torch.zeros(1, SR), SR)) == ["silent"]
    assert health_flags(audio_health(sine(440., 0.005), SR)) == ["very quiet"]
    assert health_flags(audio_health(torch.cat([sine(440.)[:, :SR // 4], torch.zeros(1, SR)], -1), SR)) == ["mostly silent"]
    assert health_flags(audio_health(sine(440.), SR)) == []
    assert health_flags(audio_health(sine(440.) + 0.1, SR), thresholds={"dc_offset": 0.5}) == []


@pytest.mark.parametrize("mode", ["online", "preprocessed"])
def test_dataset_health(tmp_path, mode):
    root = tmp_path / "data"
    root.mkdir()
    torchaudio.save(str(root / "ok.wav"), sine(440.), sample_rate=SR)
    torchaudio.save(str(root / "clipped.wav"), sine(440., 1.5).clamp(-1, 1), sample_rate=SR)
    torchaudio.save(str(root / "dc.wav"), sine(440., 0.3) + 0.1, sample_rate=SR)
    if mode == "online":
        dataset = AudioDataset(root, sr=SR, write_index=False)
    else:
        preprocess_dataset(root, out=tmp_path / "pre", chunk_length=SR // 2, sample_rate=SR, force=True)
        dataset = AudioDataset(tmp_path / "pre")
    report = dataset_health(dataset)
    flags = {r['audio_path'].split("/")[-1]: r['flags'] for r in report['files']}
    assert flags == {"ok.wav": [], "clipped.wav": ["clipping"], "dc.wav": ["dc offset"]}
    assert report['summary']['n_flagged'] == 2 and report['summary']['flags'] == {"clipping": 1, "dc offset": 1}
    assert len(dataset_health(dataset, max_files=1)['files']) == 1
