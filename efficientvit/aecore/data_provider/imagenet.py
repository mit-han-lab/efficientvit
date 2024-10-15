import os
from dataclasses import dataclass

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from efficientvit.aecore.data_provider.base import BaseDataProvider, BaseDataProviderConfig

__all__ = ["ImageNetDataProviderConfig", "ImageNetDataProvider"]


@dataclass
class ImageNetDataProviderConfig(BaseDataProviderConfig):
    name: str = "imagenet"
    data_dir: str = "~/dataset/imagenet"


class ImageNetDataProvider(BaseDataProvider):
    def __init__(self, cfg: ImageNetDataProviderConfig):
        super().__init__(cfg)
        self.cfg: ImageNetDataProviderConfig

    def build_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        transform = self.build_transform()
        train_dataset = ImageFolder(os.path.join(self.cfg.data_dir, "train"), transform)
        test_dataset = ImageFolder(os.path.join(self.cfg.data_dir, "val"), transform)
        val_dataset = None
        return train_dataset, val_dataset, test_dataset
