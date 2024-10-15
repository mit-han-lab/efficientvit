from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from efficientvit.diffusioncore.data_provider.base import BaseDataProvider, BaseDataProviderConfig

__all__ = ["SampleClassDataProviderConfig", "SampleClassDataset", "SampleClassDataProvider"]


@dataclass
class SampleClassDataProviderConfig(BaseDataProviderConfig):
    name: str = "sample_class"
    num_classes: int = 1000
    num_samples: int = 50000
    seed: int = 0


class SampleClassDataset(Dataset):
    def __init__(self, cfg: SampleClassDataProviderConfig):
        self.cfg = cfg
        self.generator = torch.Generator()
        self.generator.manual_seed(cfg.seed)
        self.class_ids = torch.randint(0, cfg.num_classes, (cfg.num_samples,), generator=self.generator).int()

    def __len__(self):
        return self.cfg.num_samples

    def __getitem__(self, index):
        return self.class_ids[index], self.cfg.num_classes


class SampleClassDataProvider(BaseDataProvider):
    def __init__(self, cfg: SampleClassDataProviderConfig):
        super().__init__(cfg)
        self.cfg: SampleClassDataProviderConfig

    def build_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        train_dataset = None
        val_dataset = SampleClassDataset(self.cfg)
        test_dataset = SampleClassDataset(self.cfg)
        return train_dataset, val_dataset, test_dataset
