import os
import sys
from dataclasses import dataclass, field

import torch
from omegaconf import MISSING, OmegaConf
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.aecore.data_provider.imagenet import ImageNetDataProvider, ImageNetDataProviderConfig
from efficientvit.apps.metrics.fid.fid import FIDStats, FIDStatsConfig
from efficientvit.apps.utils.dist import dist_init, get_dist_local_rank, is_master


@dataclass
class GenerateReferenceConfig:
    split: str = "test"
    fid: FIDStatsConfig = field(default_factory=FIDStatsConfig)

    # dataset
    dataset: str = MISSING
    imagenet: ImageNetDataProviderConfig = field(default_factory=ImageNetDataProviderConfig)


def main():
    default_cfg = OmegaConf.structured(GenerateReferenceConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg: GenerateReferenceConfig = OmegaConf.to_object(OmegaConf.merge(default_cfg, cli_cfg))

    dist_init()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(get_dist_local_rank())

    if cfg.dataset == "imagenet":
        data_provider = ImageNetDataProvider(cfg.imagenet)
    else:
        raise NotImplementedError

    fid_stats = FIDStats(cfg.fid)

    def add_data_from_dataloader(dataloader):
        for images, _ in tqdm(dataloader, disable=not is_master()):
            images = images.cuda()
            fid_stats.add_data(images)

    if "train" in cfg.split.split("_"):
        print(f"adding data from split train")
        add_data_from_dataloader(data_provider.train)
    if "valid" in cfg.split.split("_"):
        print(f"adding data from split valid")
        add_data_from_dataloader(data_provider.valid)
    if "test" in cfg.split.split("_"):
        print(f"adding data from split test")
        add_data_from_dataloader(data_provider.test)

    fid_stats.get_stats()


if __name__ == "__main__":
    main()
