import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import MISSING
from torch.utils.data import DataLoader
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure
from torchvision.utils import save_image
from tqdm import tqdm

from efficientvit.ae_model_zoo import DCAE_HF
from efficientvit.aecore.data_provider.imagenet import ImageNetDataProvider, ImageNetDataProviderConfig
from efficientvit.apps.metrics.fid.fid import FIDStats, FIDStatsConfig
from efficientvit.apps.metrics.psnr.psnr import PSNRStats, PSNRStatsConfig
from efficientvit.apps.utils.dist import (
    dist_barrier,
    dist_init,
    get_dist_local_rank,
    get_dist_rank,
    is_dist_initialized,
    is_master,
)
from efficientvit.apps.utils.metric import AverageMeter
from efficientvit.models.efficientvit.dc_ae import DCAE
from efficientvit.models.utils.network import get_dtype_from_str, is_parallel

__all__ = ["EvaluatorConfig", "Evaluator"]


time_stamp = time.time()


@dataclass
class EvaluatorConfig:
    run_dir: str = MISSING
    seed: int = 0

    evaluate_split: str = "test"
    evaluate_dir_name: Optional[str] = None
    num_save_images: int = 64
    save_images_at_all_procs: bool = False
    save_all_images: bool = False

    resolution: int = 256
    amp: str = "fp32"  # "bf16"

    # dataset
    dataset: str = MISSING
    imagenet: ImageNetDataProviderConfig = field(
        default_factory=lambda: ImageNetDataProviderConfig(resolution="${..resolution}")
    )

    # model
    model: str = MISSING

    # metrics
    compute_fid: bool = True
    fid: FIDStatsConfig = field(default_factory=FIDStatsConfig)
    compute_psnr: bool = True
    psnr: PSNRStatsConfig = field(default_factory=PSNRStatsConfig)
    compute_ssim: bool = True
    compute_lpips: bool = True


class Evaluator:
    def __init__(self, cfg: EvaluatorConfig):
        self.cfg = cfg
        self.setup_dist_env()
        self.setup_seed()

        if cfg.amp == "tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            cfg.amp = "fp32"

        # data provider
        if cfg.dataset == "imagenet":
            self.data_provider = ImageNetDataProvider(cfg.imagenet)
        else:
            raise ValueError(f"dataset {cfg.dataset} is not supported")

        # model
        model = DCAE_HF.from_pretrained(f"mit-han-lab/{cfg.model}")

        # if cfg.channels_last:
        #     model = model.to(memory_format=torch.channels_last)

        if is_dist_initialized():
            self.model = nn.parallel.DistributedDataParallel(
                model.cuda(), device_ids=[get_dist_local_rank()], find_unused_parameters=True
            )
            self.rank = get_dist_rank()
        else:
            self.model = model.cuda()
            self.rank = 0

    def setup_dist_env(self) -> None:
        dist_init()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(get_dist_local_rank())

    def setup_seed(self) -> None:
        seed = get_dist_rank() + self.cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @property
    def enable_amp(self) -> bool:
        return self.cfg.amp != "fp32"

    @property
    def amp_dtype(self) -> torch.dtype:
        return get_dtype_from_str(self.cfg.amp)

    @property
    def network(self) -> DCAE:
        return self.model.module if is_parallel(self.model) else self.model

    def run_step(self, images, global_step: int = 0):
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
            output, loss, info = self.model(images, global_step)
        return {"output": output, "loss": loss, "info": info}

    @torch.no_grad
    def evaluate_single_dataloader(
        self, dataloader: DataLoader, step: int, f_log=sys.stdout, additional_dir_name: str = ""
    ) -> dict[str, Any]:
        self.model.eval()
        valid_loss = AverageMeter(is_distributed=is_dist_initialized())
        device = torch.device("cuda")

        # metrics
        compute_fid = self.cfg.compute_fid
        fid_stats = FIDStats(self.cfg.fid)
        if self.cfg.compute_psnr:
            psnr = PSNRStats(self.cfg.psnr)
        if self.cfg.compute_ssim:
            ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 255.0)).to(device)
        if self.cfg.compute_lpips:
            lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

        if self.cfg.evaluate_dir_name is not None:
            evaluate_dir = os.path.join(self.cfg.run_dir, self.cfg.evaluate_dir_name, additional_dir_name)
        else:
            evaluate_dir = os.path.join(self.cfg.run_dir, f"{step}", additional_dir_name)
        if is_master():
            os.makedirs(evaluate_dir, exist_ok=True)
        if is_dist_initialized():
            dist_barrier()

        with tqdm(
            total=len(dataloader),
            desc="Valid Steps #{}".format(step),
            disable=not is_master(),
            file=f_log,
            mininterval=10.0,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ) as t:
            num_saved_images = 0
            for _, (images, _) in enumerate(dataloader):
                # preprocessing
                images = images.cuda()
                # if self.cfg.channels_last:
                #     images = images.to(memory_format=torch.channels_last)
                # forward
                output_dict = self.run_step(images)
                if (
                    num_saved_images < self.cfg.num_save_images and (is_master() or self.cfg.save_images_at_all_procs)
                ) or self.cfg.save_all_images:
                    device = images.device
                    input_images = images * 0.5 + 0.5
                    output_images = output_dict["output"] * 0.5 + 0.5
                    for j in range(input_images.shape[0]):
                        save_image(
                            torch.cat([input_images[j : j + 1], output_images[j : j + 1]], dim=3),
                            os.path.join(evaluate_dir, f"{self.rank}_{num_saved_images}.png"),
                        )
                        num_saved_images += 1
                        if num_saved_images >= self.cfg.num_save_images and not self.cfg.save_all_images:
                            break
                    del input_images, output_images
                # update metrics
                valid_loss.update(output_dict["loss"].item(), images.shape[0])
                if compute_fid:
                    device = output_dict["output"].device
                    output_images = output_dict["output"] * 0.5 + 0.5
                    fid_stats.add_data(output_images)
                images_ref_uint8 = (255 * ((images + 1) / 2) + 0.5).clamp(0, 255).to(torch.uint8)
                images_pred_uint8 = (255 * ((output_dict["output"] + 1) / 2) + 0.5).clamp(0, 255).to(torch.uint8)
                if self.cfg.compute_psnr:
                    psnr.add_data(images_ref_uint8, images_pred_uint8)
                if self.cfg.compute_ssim:
                    ssim.update(images_ref_uint8, images_pred_uint8)
                if self.cfg.compute_lpips:
                    lpips.update(images_ref_uint8 / 255, images_pred_uint8 / 255)
                ## tqdm
                postfix_dict = {
                    "loss": valid_loss.avg,
                    "bs": images.shape[0],
                    "res": images.shape[2],
                }
                t.set_postfix(postfix_dict, refresh=False)
                t.update()
        valid_info_dict = {
            "loss": valid_loss.avg,
        }
        torch.cuda.empty_cache()
        if compute_fid:
            valid_info_dict["fid"] = fid_stats.compute_fid()
        if self.cfg.compute_psnr:
            valid_info_dict["psnr"] = psnr.compute()
        if self.cfg.compute_ssim:
            valid_info_dict["ssim"] = ssim.compute().item()
        if self.cfg.compute_lpips:
            valid_info_dict["lpips"] = lpips.compute().item()
        return valid_info_dict

    @torch.no_grad
    def evaluate(self, step: int, f_log=sys.stdout) -> dict[str, Any]:
        if self.cfg.evaluate_split == "train":
            dataloader = self.data_provider.train
        elif self.cfg.evaluate_split == "valid":
            dataloader = self.data_provider.valid
        elif self.cfg.evaluate_split == "test":
            dataloader = self.data_provider.test
        else:
            raise NotImplementedError
        valid_info_dict = self.evaluate_single_dataloader(dataloader, step, f_log)
        return valid_info_dict
