"""Deprecated : balanced and curriculum sampling are implemented in acids_dataset.sampling (WeightedItemSampler, 
BalancedBatchSampler, Curriculum), and available through AudioDataset.get_sampler / get_loader (balance_by, curriculum, 
balanced_batches)."""
from ..sampling import WeightedItemSampler, BalancedBatchSampler, Curriculum, LossTracker, balance_weights


class BalancedSampler():
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("BalancedSampler is replaced by acids_dataset.sampling ; use dataset.get_loader(balance_by=..., balanced_batches=...)")
