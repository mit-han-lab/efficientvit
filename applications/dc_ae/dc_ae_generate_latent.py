import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import MISSING, OmegaConf
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.ae_model_zoo import DCAE_HF, REGISTERED_DCAE_MODEL, AutoencoderKL
from efficientvit.apps.data_provider.sampler import DistributedRangedSampler
from efficientvit.apps.utils.dist import (
    dist_barrier,
    dist_init,
    get_dist_local_rank,
    get_dist_rank,
    get_dist_size,
    is_master,
)
from efficientvit.apps.utils.image import CustomImageFolder, DMCrop
from efficientvit.models.utils.network import get_dtype_from_str


@dataclass
class GenerateLatentConfig:
    image_root_path: str = MISSING
    latent_root_path: str = MISSING
    results_path: Optional[str] = None
    resolution: int = MISSING

    model_name: str = MISSING
    dtype: str = "fp32"
    scaling_factor: Optional[float] = None

    batch_size: int = 64
    num_workers: int = 8

    task_id: int = 0
    num_samples_per_task: Optional[int] = None
    resume: bool = True


def image_path_to_latent_path(image_path: str, image_root_path: str, latent_root_path: str) -> str:
    relative_image_path = os.path.relpath(image_path, image_root_path)
    last_dot_pos = relative_image_path.rfind(".")
    relative_npz_path = relative_image_path[:last_dot_pos] + ".npy"
    latent_path = os.path.join(latent_root_path, relative_npz_path)
    return latent_path


def main():
    torch.set_grad_enabled(False)
    cfg: GenerateLatentConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(GenerateLatentConfig), OmegaConf.from_cli())
    )

    dist_init()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(get_dist_local_rank())

    device = torch.device("cuda")
    dtype = get_dtype_from_str(cfg.dtype)
    if cfg.model_name in REGISTERED_DCAE_MODEL:
        model = DCAE_HF.from_pretrained(f"mit-han-lab/{cfg.model_name}").to(device=device, dtype=dtype)
        assert cfg.scaling_factor is not None
    elif cfg.model_name in ["stabilityai/sd-vae-ft-ema", "flux-vae"]:
        model = AutoencoderKL(cfg.model_name).to(device=device, dtype=dtype)
        cfg.scaling_factor = model.model.config.scaling_factor
    else:
        raise ValueError(f"{cfg.model} is not supported for generating latent")

    transform = transforms.Compose(
        [
            DMCrop(cfg.resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5, inplace=True),
        ]
    )

    dataset = CustomImageFolder(cfg.image_root_path, transform, return_dict=True)

    if cfg.num_samples_per_task is not None:
        num_tasks = (len(dataset) - 1) // cfg.num_samples_per_task + 1
        print(f"num_tasks {num_tasks}")
        start, end = min(cfg.num_samples_per_task * cfg.task_id, len(dataset)), min(
            cfg.num_samples_per_task * (cfg.task_id + 1), len(dataset)
        )
        indices = list(range(start, end))
        dataset = torch.utils.data.Subset(dataset, indices)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        sampler=DistributedRangedSampler(dataset, num_replicas=get_dist_size(), rank=get_dist_rank(), shuffle=False),
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    if is_master():
        os.makedirs(cfg.latent_root_path, exist_ok=cfg.resume or cfg.num_samples_per_task is not None)
    dist_barrier()

    for batch_idx, input_dict in tqdm(enumerate(data_loader), total=len(data_loader), disable=not is_master()):
        skip = False
        if cfg.resume:
            skip = True
            for image_path in input_dict["image_path"]:
                latent_path = image_path_to_latent_path(image_path, cfg.image_root_path, cfg.latent_root_path)
                try:
                    data = np.load(latent_path)
                    assert data.shape[1] == data.shape[2] == cfg.resolution // model.spatial_compression_ratio
                except Exception:
                    skip = False
        if skip:
            if is_master():
                print(f"skip batch {batch_idx}")
            continue

        images = input_dict["image"].cuda()
        latents = model.encode(images)
        latents = latents * cfg.scaling_factor
        for i, (image_path, _) in enumerate(zip(input_dict["image_path"], input_dict["label"])):
            latent = latents[i].cpu().numpy()
            latent_path = image_path_to_latent_path(image_path, cfg.image_root_path, cfg.latent_root_path)
            os.makedirs(os.path.dirname(latent_path), exist_ok=True)
            np.save(latent_path, latent)

    dist_barrier()

    if cfg.results_path is not None and is_master():
        os.makedirs(cfg.results_path, exist_ok=True)
        with open(os.path.join(cfg.results_path, f"{cfg.num_samples_per_task}_{cfg.task_id}.txt"), "w") as f:
            f.write("complete!")


if __name__ == "__main__":
    main()
