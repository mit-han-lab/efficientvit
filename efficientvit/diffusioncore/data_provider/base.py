from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING
from torch.utils.data import DataLoader, Dataset

from efficientvit.apps.data_provider.sampler import DistributedRangedSampler
from efficientvit.apps.utils.dist import get_dist_rank, get_dist_size, is_master

__all__ = ["BaseDataProviderConfig", "BaseDataProvider"]


@dataclass
class BaseDataProviderConfig:
    name: str = MISSING
    batch_size: int = 32
    n_worker: int = 8
    train_drop_last: bool = True
    seed: int = 0


class BaseDataProvider:
    def __init__(self, cfg: BaseDataProviderConfig):
        self.cfg = cfg
        self.train_dataset, self.val_dataset, self.test_dataset = self.build_datasets()
        if is_master():
            try:
                print(f"len(train_dataset)={len(self.train_dataset)}" if self.train_dataset is not None else 0)
                print(f"len(val_dataset)={len(self.val_dataset) if self.val_dataset is not None else 0}")
                print(f"len(test_dataset)={len(self.test_dataset) if self.test_dataset is not None else 0}")
            except Exception as e:
                print(f"can not print len(dataset): {e}")
        self.num_replicas = get_dist_size()
        self.rank = get_dist_rank()
        self.train = self.build_dataloader(
            self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=cfg.train_drop_last
        )
        self.valid = self.build_dataloader(
            self.val_dataset, batch_size=self.cfg.batch_size, shuffle=False, drop_last=False
        )
        self.test = self.build_dataloader(
            self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False, drop_last=False
        )

    def build_datasets(self) -> tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        raise NotImplementedError

    def build_dataloader(
        self, dataset: Optional[Dataset], batch_size: int, shuffle: bool, drop_last: bool
    ) -> Optional[DataLoader]:
        if dataset is None:
            return None
        sampler = DistributedRangedSampler(
            dataset, self.num_replicas, self.rank, shuffle=shuffle, seed=self.cfg.seed, drop_last=drop_last
        )
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.cfg.n_worker,
            pin_memory=True,
            drop_last=drop_last,
        )

    def set_epoch(self, epoch: int) -> None:
        self.train.sampler.set_epoch(epoch)

    def set_batch_idx(self, batch_idx: int) -> None:
        self.train.sampler.set_iter_idx(batch_idx * self.cfg.batch_size)
