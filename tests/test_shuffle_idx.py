"""Persistent index shuffling (shuffle_idx) : deterministic permutation of items, consistent across copies, workers,
labels and combined datasets."""
import pickle
import pytest
import torch
import torchaudio

from acids_dataset import preprocess_dataset
from acids_dataset.datasets import AudioDataset, CombinedAudioDataset

SR = 8000
COUNTS = {"piano": 6, "guitar": 4}
REGEXP = ["**/{{instrument}}_*.wav"]


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    root = tmp_path_factory.mktemp("shuffle_idx")
    g = torch.Generator().manual_seed(0)
    (root / "raw").mkdir()
    for inst, n in COUNTS.items():
        for i in range(n):
            torchaudio.save(str(root / "raw" / f"{inst}_{i}.wav"), 0.1 * torch.randn(1, SR, generator=g), sample_rate=SR)
    preprocess_dataset(root / "raw", out=root / "pre", chunk_length=SR, sample_rate=SR, channels=1, meta_regexp=REGEXP, force=True)
    return {"online": root / "raw", "preprocessed": root / "pre"}


def load(paths, mode, **kwargs):
    if mode == "online":
        # one chunk per file, taken at its beginning : items are deterministic
        return AudioDataset(paths["online"], sr=SR, chunk_length=SR, meta_regexp=REGEXP, write_index=False, **kwargs)
    return AudioDataset(paths["preprocessed"], **kwargs)


@pytest.mark.parametrize("mode", ["online", "preprocessed"])
def test_shuffle_idx_permutes_items(paths, mode):
    plain = load(paths, mode)
    shuffled = load(paths, mode, shuffle_idx=3)
    assert len(shuffled) == len(plain) and shuffled.shuffle_idx == 3 and plain.shuffle_idx is None
    perm = shuffled._index_permutation()
    assert sorted(perm.tolist()) == list(range(len(plain))) and perm.tolist() != list(range(len(plain)))
    for i in range(len(plain)):
        assert torch.equal(shuffled[i], plain[int(perm[i])])
    # negative indices and multi-indices follow the permutation
    assert torch.equal(shuffled[-1], plain[int(perm[-1])])
    assert torch.equal(shuffled[[0, 1]], torch.stack([plain[int(perm[0])], plain[int(perm[1])]]))
    with pytest.raises(IndexError):
        shuffled[len(plain)]


@pytest.mark.parametrize("mode", ["online", "preprocessed"])
def test_shuffle_idx_is_persistent(paths, mode):
    a, b = load(paths, mode, shuffle_idx=7), load(paths, mode, shuffle_idx=7)
    assert a._index_permutation().tolist() == b._index_permutation().tolist()
    assert load(paths, mode, shuffle_idx=8)._index_permutation().tolist() != a._index_permutation().tolist()
    # random seeds are kept by copies (e.g. in workers)
    r = load(paths, mode, shuffle_idx=True)
    assert isinstance(r.shuffle_idx, int)
    assert pickle.loads(pickle.dumps(r))._index_permutation().tolist() == r._index_permutation().tolist()


@pytest.mark.parametrize("mode", ["online", "preprocessed"])
def test_shuffle_idx_labels_follow_items(paths, mode):
    plain = load(paths, mode)
    shuffled = load(paths, mode, shuffle_idx=1)
    perm = shuffled._index_permutation()
    plain_labels = plain.item_labels("instrument")
    assert shuffled.item_labels("instrument") == [plain_labels[i] for i in perm]


@pytest.mark.parametrize("mode", ["online", "preprocessed"])
def test_shuffle_idx_with_workers(paths, mode):
    shuffled = load(paths, mode, shuffle_idx=5)
    loader = shuffled.get_loader(batch_size=2, shuffle=False, num_workers=2)
    batches = torch.cat(list(loader))
    expected = torch.stack([shuffled[i] for i in range(len(shuffled))])
    assert torch.equal(batches, expected)


def test_shuffle_idx_combined(paths):
    datasets = [load(paths, "online"), load(paths, "preprocessed")]
    plain = CombinedAudioDataset(datasets)
    shuffled = CombinedAudioDataset(datasets, shuffle_idx=2)
    perm = shuffled._index_permutation()
    for i in range(len(plain)):
        assert torch.equal(shuffled[i], plain[int(perm[i])])
    plain_weights = plain.sampling_weights()
    assert torch.equal(shuffled.sampling_weights(), plain_weights[torch.as_tensor(perm)])


@pytest.mark.parametrize("mode", ["online", "preprocessed"])
def test_shuffle_idx_set_afterwards(paths, mode):
    plain = load(paths, mode)
    reference = load(paths, mode, shuffle_idx=4)
    ds = load(paths, mode)
    ds.shuffle_idx = 4
    assert ds._index_permutation().tolist() == reference._index_permutation().tolist()
    assert torch.equal(ds[0], reference[0])
    ds.shuffle_idx = None
    assert ds.shuffle_idx is None and torch.equal(ds[0], plain[0])
    # configure returns a shuffled copy, without modifying the dataset
    configured = ds.configure(shuffle_idx=4)
    assert configured is not ds and ds.shuffle_idx is None and configured.shuffle_idx == 4
    assert configured.configure(shuffle_idx=4).shuffle_idx == 4
    assert configured.configure(shuffle_idx=False).shuffle_idx is None
    # get_loader gives loading arguments to configure
    batches = torch.cat(list(ds.get_loader(batch_size=2, shuffle=False, shuffle_idx=4)))
    assert torch.equal(batches, torch.stack([reference[i] for i in range(len(reference))]))


@pytest.mark.parametrize("mode", ["online", "preprocessed"])
def test_shuffle_idx_split(paths, mode):
    # partitions do not depend on shuffle_idx, and are shuffled with the seed of the split dataset
    split_kwargs = dict(seed=0, use_meta_if_available=False)
    plain_parts = load(paths, mode).split({"train": 0.6, "valid": 0.4}, **split_kwargs)
    shuffled_parts = load(paths, mode, shuffle_idx=6).split({"train": 0.6, "valid": 0.4}, **split_kwargs)
    for name, part in shuffled_parts.items():
        plain = plain_parts[name]
        assert len(part) == len(plain) and part.shuffle_idx == 6
        perm = part._index_permutation()
        assert sorted(perm.tolist()) == list(range(len(plain)))
        for i in range(len(part)):
            assert torch.equal(part[i], plain[int(perm[i])])
        plain_labels = plain.item_labels("instrument")
        assert part.item_labels("instrument") == [plain_labels[j] for j in perm]
    # a partition can be shuffled independently of its parent
    part = plain_parts["train"]
    part.shuffle_idx = 6
    assert torch.equal(part[0], shuffled_parts["train"][0])


def test_shuffle_idx_combined_configure(paths):
    datasets = [load(paths, "online"), load(paths, "preprocessed")]
    combined = CombinedAudioDataset(datasets).configure(shuffle_idx=2)
    assert combined.shuffle_idx == 2 and all(d.shuffle_idx is None for d in combined.datasets)
