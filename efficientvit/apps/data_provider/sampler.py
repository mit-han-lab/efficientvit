import torch
from torch.utils.data import Dataset, Sampler

__all__ = ["DistributedRangedSampler"]


class DistributedRangedSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        assert rank >= 0 and rank < num_replicas
        self.num_samples = len(dataset)
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        if drop_last:
            self.num_samples_per_rank = self.num_samples // num_replicas
        else:
            self.num_samples_per_rank = (self.num_samples - 1) // num_replicas + 1
        self.epoch = 0
        self.iter_idx = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.iter_idx = 0

    def set_iter_idx(self, iter_idx):
        self.iter_idx = iter_idx

    def __len__(self):
        return self.num_samples_per_rank

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_samples, generator=g).tolist()
            if not self.drop_last:
                total_size = self.num_replicas * self.num_samples_per_rank
                padding_size = total_size - len(indices)
                indices += (indices * ((padding_size - 1) // len(indices) + 1))[:padding_size]
            indices = indices[self.rank * self.num_samples_per_rank : (self.rank + 1) * self.num_samples_per_rank]
            assert len(indices) == self.num_samples_per_rank
            yield from indices[self.iter_idx :]
        else:
            start = self.rank * self.num_samples_per_rank + self.iter_idx
            end = (self.rank + 1) * self.num_samples_per_rank
            indices = torch.arange(self.num_replicas * self.num_samples_per_rank) % self.num_samples
            yield from indices[start:end]
