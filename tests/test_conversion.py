"""Conversion of online datasets into pre-processed datasets (OnlineAudioDataset.preprocess)."""
import math
import pytest
import torch
import torchaudio
import yaml

from acids_dataset.datasets import AudioDataset, OnlineAudioDataset, PreprocessedAudioDataset
from acids_dataset.features import Loudness, SpectralCentroid

SR = 22050
REGEXP = ["**/{{instrument}}_*.wav"]


def write(path, x, sr=SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), x, sample_rate=sr)


@pytest.fixture
def raw(tmp_path):
    g = torch.Generator().manual_seed(0)
    for inst, n in [("piano", 4), ("guitar", 3)]:
        for i in range(n):
            write(tmp_path / "raw" / f"{inst}_{i}.wav", 0.2 * torch.randn(1, (2 + i) * SR, generator=g))
    write(tmp_path / "raw" / "excluded" / "guitar_9.wav", 0.2 * torch.randn(1, SR, generator=g))
    (tmp_path / "raw" / "broken.wav").write_bytes(b"not audio")
    return tmp_path


def online(raw, **kwargs):
    kwargs = {"sr": SR, "channels": 1, "chunk_length": SR, "meta_regexp": REGEXP, "exclude": ["excluded/*"], "write_index": False,
              "features": [Loudness(sr=SR, keep_channels=False)], "output_pattern": "{waveform,loudness}", **kwargs}
    return AudioDataset(raw / "raw", **kwargs)


def metadata(path):
    with open(path / "metadata.yaml") as f:
        return yaml.safe_load(f)


def test_conversion(raw):
    dataset = online(raw)
    pre = dataset.preprocess(raw / "compiled")
    assert type(pre) is PreprocessedAudioDataset
    # same files (filters, exclusions and unreadable files of the online dataset)
    assert sorted(f['audio_path'] for f in pre.files) == sorted(f['audio_path'] for f in dataset.files)
    md = metadata(raw / "compiled")
    assert md['sr'] == SR and md['chunk_length'] == pytest.approx(1.) and md['hop_length'] == pytest.approx(0.5) and md['channels'] == 1
    assert md['source']['type'] == "online" and md['source']['exclude'] == ["excluded/*"] and md['source']['meta_regexp'] == REGEXP
    assert any(p.endswith("broken.wav") for p in md['source']['invalid_files'])
    # features and regexp values
    assert set(pre.features.keys()) == {"instrument", "loudness"}
    assert set(pre.feature_hash["instrument"].keys()) == {"piano", "guitar"}
    item = pre[0]
    assert set(item.keys()) == {"waveform", "loudness"} and item["waveform"].shape == (1, SR)
    # feature values match the ones of the online dataset on the same chunks
    for file_idx, file in enumerate(dataset.files):
        online_value = dataset.get_fragment(file_idx, start=0., length=SR).get("loudness")
        keys = [pre._key_to_str(k) for k in pre.feature_hash["original_path"][file['audio_path']]]
        first = next(k for k in keys if float(pre.loader[k].get_metadata()['start_pos']) == 0.)
        assert abs(float(pre.loader[first].get("loudness")) - float(online_value)) < 1e-2


def test_conversion_parameters(raw):
    dataset = online(raw, pad_mode="replicate", features=[SpectralCentroid(sr=SR)])
    pre = dataset.preprocess(raw / "compiled", hop_length=SR, features=[Loudness(sr=SR, keep_channels=False)], waveform=False,
                             output_pattern="loudness")
    md = metadata(raw / "compiled")
    assert md['hop_length'] == pytest.approx(1.) and md['waveform'] is False and md['source']['pad_mode'] == "replicate"
    assert set(pre.features.keys()) == {"instrument", "loudness"}
    # non-overlapping chunks : one chunk per second of audio (last chunks are padded)
    assert len(pre) == sum(math.ceil(f['duration']) for f in dataset.files)
    assert pre[0].shape == ()
    with pytest.raises(ValueError):
        dataset.preprocess(raw / "compiled")


def test_partitions(raw):
    dataset = online(raw)
    dataset.split({"train": 0.7, "valid": 0.3}, features=["instrument"], seed=0, write="by_instrument")
    dataset.split({"train": 0.5, "valid": 0.5}, seed=1, write="random")
    pre = dataset.preprocess(raw / "all")
    for name in ["by_instrument", "random"]:
        online_partitions, pre_partitions = dataset.load_partition(name), pre.load_partition(name)
        for p in online_partitions:
            online_files = {dataset.files[i]['audio_path'] for i in online_partitions[p].file_indices}
            keys = set(map(pre._key_to_str, pre_partitions[p]._subindices))
            pre_files = {a for a, ks in pre.feature_hash["original_path"].items() if keys & set(map(pre._key_to_str, ks))}
            assert pre_files == online_files
            # every chunk of the files is in the partition
            assert len(keys) == sum(len(pre.feature_hash["original_path"][a]) for a in online_files)
    assert not dataset.preprocess(raw / "none", partitions=None).has_partition("random")
    only = dataset.preprocess(raw / "only", partitions=["random"])
    assert only.has_partition("random") and not only.has_partition("by_instrument")


def test_partition_conversion(raw):
    # a partition of an online dataset gives a pre-processed dataset of its files
    train = online(raw).split({"train": 0.5, "valid": 0.5}, seed=0)["train"]
    pre = train.preprocess(raw / "train")
    assert sorted(f['audio_path'] for f in pre.files) == sorted(train.files[i]['audio_path'] for i in train.file_indices)


def test_health_exclusion(tmp_path):
    t = torch.arange(SR) / SR
    write(tmp_path / "raw" / "ok.wav", 0.5 * torch.sin(2 * math.pi * 440 * t)[None])
    write(tmp_path / "raw" / "silent.wav", torch.zeros(1, SR))
    write(tmp_path / "raw" / "clipped.wav", (2 * torch.sin(2 * math.pi * 440 * t)).clamp(-1, 1)[None])
    dataset = AudioDataset(tmp_path / "raw", sr=SR, chunk_length=SR // 2, write_index=False)
    pre = dataset.preprocess(tmp_path / "compiled", exclude_flags=["silent", "clipping"])
    assert [f['audio_path'] for f in pre.files] == ["raw/ok.wav"]
    assert sorted(metadata(tmp_path / "compiled")['source']['excluded_files']) == ["raw/clipped.wav", "raw/silent.wav"]
    with pytest.raises(ValueError):
        AudioDataset(tmp_path / "raw", sr=SR, chunk_length=SR, write_index=False).preprocess(tmp_path / "none", exclude_flags=["silent", "clipping", "very quiet"],
                                                                                          health_thresholds={"peak_db": 10.})


def test_loudness_threshold(tmp_path):
    x = torch.zeros(1, 4 * SR)
    x[:, :SR] = 0.2 * torch.randn(1, SR)
    write(tmp_path / "raw" / "a.wav", x)
    plain = AudioDataset(tmp_path / "raw", sr=SR, chunk_length=SR, write_index=False).preprocess(tmp_path / "plain", hop_length=SR)
    filtered = AudioDataset(tmp_path / "raw", sr=SR, chunk_length=SR, loudness_threshold=-50, write_index=False).preprocess(tmp_path / "filtered", hop_length=SR)
    # silent chunks are discarded in pre-processing, as they are drawn again online
    assert len(plain) == 4 and len(filtered) == 1


def test_errors_and_folders(tmp_path):
    write(tmp_path / "a" / "x.wav", 0.1 * torch.randn(1, SR), sr=SR)
    write(tmp_path / "b" / "y.wav", 0.1 * torch.randn(1, 2 * SR), sr=SR)
    with pytest.raises(ValueError):
        OnlineAudioDataset(tmp_path / "a", write_index=False).preprocess(tmp_path / "out")
    dataset = OnlineAudioDataset([tmp_path / "a", tmp_path / "b"], sr=SR, chunk_length=SR, write_index=False)
    pre = dataset.preprocess(tmp_path / "out")
    assert sorted(f['audio_path'] for f in pre.files) == ["a/x.wav", "b/y.wav"]
    assert metadata(tmp_path / "out")['source']['paths'] == [str((tmp_path / "a").resolve()), str((tmp_path / "b").resolve())]
