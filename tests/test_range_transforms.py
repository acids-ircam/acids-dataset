"""Peak feature, and CompressRange / NormRange transforms."""
import numpy as np
import pytest
import torch

from acids_dataset.features import Peak
from acids_dataset.transforms import CompressRange, NormRange
from acids_dataset.utils import loudness

SR = 22050


def test_peak():
    x = torch.zeros(2, 1000)
    x[0, 10], x[1, 20] = -0.5, 0.25
    assert torch.isclose(Peak(sr=SR)(x), torch.tensor(0.5))
    assert torch.allclose(Peak(keep_channels=True, sr=SR)(x), torch.tensor([0.5, 0.25]))
    assert torch.isclose(Peak(db=True, sr=SR)(x), torch.tensor(20 * np.log10(0.5), dtype=torch.float32), atol=1e-4)
    # batches, numpy inputs
    batch = torch.stack([x, 2 * x])
    assert torch.allclose(Peak(sr=SR).batch_call(batch), torch.tensor([0.5, 1.]))
    assert torch.isclose(Peak(sr=SR)(x.numpy()), torch.tensor(0.5))


def test_norm_range():
    x = torch.randn(3, 2, 1000)
    x[1] *= 10
    x[2] = 0
    out = NormRange(max_value=0.8)(x)
    peaks = out.abs().amax(dim=(-2, -1))
    assert torch.allclose(peaks[:2], torch.tensor([0.8, 0.8]))
    assert torch.equal(out[2], x[2])
    # without boost, only inputs above max_value are divided
    quiet = 0.1 * x[0] / x[0].abs().max()
    assert torch.allclose(NormRange(max_value=0.8, boost=False)(quiet), quiet)
    assert torch.isclose(NormRange(max_value=0.8, boost=False)(x[1]).abs().max(), torch.tensor(0.8))
    # numpy and 1-d inputs
    y = NormRange(max_value=0.5)(np.random.randn(500))
    assert isinstance(y, np.ndarray) and np.isclose(np.abs(y).max(), 0.5)


def frames_db(x, frame_duration=0.1):
    return loudness(x, SR, frame_length=int(frame_duration * SR))


def signal(levels_db, frame_duration=0.1, channels=1):
    """noise whose loudness steps through levels_db, one segment of 4 frames per level."""
    g = torch.Generator().manual_seed(0)
    n = int(frame_duration * SR) * 4
    segments = [torch.randn(channels, n, generator=g) * 10 ** (l / 20) for l in levels_db]
    return torch.cat(segments, -1)


def test_compress_range_measured():
    x = signal([-50, -40, -30, -20, -10])
    db_in = frames_db(x)
    assert db_in.max() - db_in.min() > 35
    out = CompressRange(target_range=(-25, -15), sr=SR)(x)
    db_out = frames_db(out)
    # loudness of frames is within the target range (up to interpolation between frames)
    assert db_out.min() > -27 and db_out.max() < -13
    assert db_out.max() - db_out.min() < 14
    # loudness order is kept
    segment_db = db_out.reshape(5, 4).mean(-1)
    assert torch.all(segment_db.diff() > 0)


def test_compress_range_given_input_range():
    x = signal([-40, -30, -20])
    out = CompressRange(target_range=(-20, -20), input_range=(-60, 0), sr=SR)(x)
    # frames at the center of segments (gains are interpolated between segments)
    db_out = frames_db(out).reshape(3, 4)[:, 1:3]
    assert torch.allclose(db_out, torch.full_like(db_out, -20.), atol=0.5)
    # input range (-60, 0) mapped linearly to (-30, -10) : ratio 1/3 around -30 -> -20
    out = CompressRange(target_range=(-30, -10), input_range=(-60, 0), sr=SR)(x)
    db_in = frames_db(x).reshape(3, 4)[:, 1:3].mean(-1)
    db_out = frames_db(out).reshape(3, 4)[:, 1:3].mean(-1)
    assert torch.allclose(db_out, -20 + (db_in + 30) / 3, atol=0.5)


def test_compress_range_shapes():
    t = CompressRange(sr=SR)
    stereo = signal([-40, -20], channels=2)
    out = t(stereo)
    assert out.shape == stereo.shape
    # same gain on every channel
    assert torch.allclose(out[0] / stereo[0], out[1] / stereo[1], atol=1e-4)
    batch = torch.stack([stereo, 0.1 * stereo])
    out = t(batch)
    assert out.shape == batch.shape
    # each item is compressed independently : both reach the same range
    assert torch.allclose(frames_db(out[0]), frames_db(out[1]), atol=0.5)
    # silence is unchanged ; short and numpy inputs
    assert torch.equal(t(torch.zeros(1, 5000)), torch.zeros(1, 5000))
    assert t(torch.randn(1, 100) * 0.1).shape == (1, 100)
    assert isinstance(t(stereo.numpy()), np.ndarray)
    assert out.abs().max() <= 1.


def sine(freq, amp=0.5, n=SR, channels=1):
    t = torch.arange(n) / SR
    return (amp * torch.sin(2 * torch.pi * freq * t))[None].repeat(channels, 1)


def test_loudness_meter_values():
    from acids_dataset.features import LoudnessMeter
    # sine of amplitude a : RMS level 20 log10(a / sqrt(2)) ; A-weighting is 0 dB at 1 kHz, about -19 dB at 100 Hz
    x = torch.cat([sine(1000, 0.5), sine(1000, 0.05)], -1)
    meter = LoudnessMeter(sr=SR, frame_length=1000)
    out = meter(x)
    assert meter.output_names == ["min_db", "max_db", "min_dba", "max_dba"]
    expected = torch.tensor([20 * np.log10(0.05 / np.sqrt(2)), 20 * np.log10(0.5 / np.sqrt(2))], dtype=torch.float32)
    assert out.shape == (4,)
    assert torch.allclose(out[:2], expected, atol=0.1)
    assert torch.allclose(out[2:], expected, atol=0.5)
    low = LoudnessMeter(properties=["max_db", "max_dba"], sr=SR, frame_length=2205)(sine(100))
    assert torch.isclose(low[0] - low[1], torch.tensor(19.1), atol=0.5)
    # silence is floored
    assert LoudnessMeter(properties="max_db", sr=SR)(torch.zeros(1, 4096)) == -120.


def test_loudness_meter_quantiles_and_shapes():
    from acids_dataset.features import LoudnessMeter
    levels = [0.01, 0.1, 0.2, 0.5]
    x = torch.cat([sine(1000, a, n=4410) for a in levels], -1)
    meter = LoudnessMeter(properties=["min_db", "db_quantiles", "dba_quantiles", "max_db"], n_quantiles=5, frame_length=441, sr=SR)
    out = meter(x)
    assert out.shape == (1 + 5 + 5 + 1,) and len(meter.output_names) == out.shape[-1]
    q = out[1:6]
    assert torch.all(q.diff() >= 0)
    assert torch.isclose(q[0], out[0]) and torch.isclose(q[-1], out[-1])
    # channels and batches
    stereo = torch.stack([x[0], 0.5 * x[0]])
    m = LoudnessMeter(keep_channels=True, frame_length=441, sr=SR)
    per_channel = m(stereo)
    assert per_channel.shape == (2, 4)
    assert torch.allclose(per_channel[0, :2] - per_channel[1, :2], torch.tensor([6.02, 6.02]), atol=0.05)
    batch = torch.stack([stereo, stereo])
    assert m.batch_call(batch).shape == (2, 2, 4)
    assert LoudnessMeter(sr=SR).batch_call(batch).shape == (2, 4)
    with pytest.raises(ValueError):
        LoudnessMeter(properties=["mean_db"])


def test_loudness_meter_lufs():
    from acids_dataset.features import LoudnessMeter
    x = torch.cat([sine(1000, 0.5, n=4410), sine(1000, 0.05, n=4410)], -1)
    meter = LoudnessMeter(properties=["min_lufs", "max_lufs", "lufs_quantiles"], n_quantiles=3, frame_length=2205, sr=SR)
    out = meter(x)
    assert out.shape == (5,) and meter.output_names[:2] == ["min_lufs", "max_lufs"]
    # LUFS of frames match utils.loudness of the same frames
    expected = loudness(x[:, :2205], SR)
    assert torch.isclose(out[1], expected, atol=0.05)
    assert torch.isclose(out[1] - out[0], torch.tensor(20.), atol=0.1)
    # combined loudness of stereo is 3 dB above mono ; per channel with keep_channels
    stereo = x.repeat(2, 1)
    assert torch.isclose(meter(stereo)[1] - out[1], torch.tensor(3.01), atol=0.05)
    per_channel = LoudnessMeter(properties=["max_lufs"], frame_length=2205, keep_channels=True, sr=SR)(stereo)
    assert per_channel.shape == (2, 1) and torch.allclose(per_channel[:, 0], out[1].expand(2), atol=0.05)
    # silence is floored, batches
    assert meter(torch.zeros(1, 4410))[0] == -120.
    assert meter.batch_call(torch.stack([stereo, stereo])).shape == (2, 5)


@pytest.mark.parametrize("frame_length", [None, 2205])
@pytest.mark.parametrize("keep_channels", [False, True])
def test_loudness_modes(frame_length, keep_channels):
    from acids_dataset.features import Loudness
    from acids_dataset.features.loudness import LoudnessScript
    x = torch.stack([sine(1000, 0.5)[0], sine(100, 0.25)[0]])[None].repeat(3, 1, 1)   # (3, 2, T)
    shapes = {m: Loudness(mode=m, frame_length=frame_length, keep_channels=keep_channels, sr=SR)(x).shape for m in ["lufs", "db", "dba"]}
    assert shapes["db"] == shapes["lufs"] == shapes["dba"]
    for mode in ["db", "dba"]:
        feature = Loudness(mode=mode, frame_length=frame_length, keep_channels=keep_channels, sr=SR)
        out = feature(x)
        scripted = torch.jit.script(feature.script())(x)
        assert torch.allclose(out, scripted, atol=1e-3), mode
        assert feature.feature_name == "loudness_%s"%mode and "mode=%s"%mode in repr(feature)
    if keep_channels:
        db = Loudness(mode="db", frame_length=frame_length, keep_channels=True, sr=SR)(x)
        dba = Loudness(mode="dba", frame_length=frame_length, keep_channels=True, sr=SR)(x)
        # sines of amplitude a : 20 log10(a / sqrt(2)) dBFS ; A-weighting is 0 dB at 1 kHz, about -19 dB at 100 Hz
        expected = torch.tensor([20 * np.log10(0.5 / np.sqrt(2)), 20 * np.log10(0.25 / np.sqrt(2))], dtype=torch.float32)
        db0 = db[0] if frame_length is None else db[0, :, 3]
        dba0 = dba[0] if frame_length is None else dba[0, :, 3]
        assert torch.allclose(db0, expected, atol=0.1)
        assert torch.isclose(dba0[0], db0[0], atol=0.2) and torch.isclose(db0[1] - dba0[1], torch.tensor(19.1), atol=0.5)
    # LUFS representation and name are unchanged ; silence is floored
    assert repr(Loudness(sr=SR)) == "Loudness(sr=%d)"%SR and Loudness(sr=SR).feature_name == "loudness"
    assert torch.all(Loudness(mode="dba", sr=SR)(torch.zeros(1, 4410)) == -70.)
    with pytest.raises(ValueError):
        Loudness(mode="sone")


@pytest.mark.parametrize("mode", ["db", "dba"])
def test_compress_range_modes(mode):
    from acids_dataset.utils import level
    x = signal([-50, -40, -30, -20, -10])
    out = CompressRange(target_range=(-25, -15), mode=mode, sr=SR)(x)
    db_out = level(out, SR, mode=mode, frame_length=int(0.1 * SR)).reshape(5, 4)[:, 1:3]
    assert db_out.min() > -26 and db_out.max() < -14
    assert torch.all(db_out.mean(-1).diff() > 0)
    with pytest.raises(ValueError):
        CompressRange(mode="sone")
