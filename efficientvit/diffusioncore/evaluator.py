import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import MISSING
from PIL import Image
from tqdm import tqdm

from efficientvit.ae_model_zoo import DCAE_HF, REGISTERED_DCAE_MODEL, AutoencoderKL
from efficientvit.apps.metrics.fid.fid import FIDStats, FIDStatsConfig
from efficientvit.apps.metrics.inception_score.inception_score import InceptionScoreStats, InceptionScoreStatsConfig
from efficientvit.apps.utils.dist import (
    dist_barrier,
    dist_init,
    get_dist_local_rank,
    get_dist_rank,
    get_dist_size,
    is_dist_initialized,
    is_master,
)
from efficientvit.diffusioncore.data_provider.sample_class import SampleClassDataProvider, SampleClassDataProviderConfig
from efficientvit.diffusioncore.models.dit import DiT, DiTConfig
from efficientvit.diffusioncore.models.uvit import UViT, UViTConfig
from efficientvit.models.utils.network import get_dtype_from_str, is_parallel

__all__ = ["EvaluatorConfig", "Evaluator"]


@dataclass
class EvaluatorConfig:
    run_dir: str = MISSING
    seed: int = 0
    allow_tf32: bool = True

    resolution: int = 512
    amp: str = "fp32"
    cfg_scale: float = 1.0
    evaluate_split: str = "test"
    evaluate_dir_name: Optional[str] = None
    num_save_images: int = 64
    save_all_images: bool = False
    save_images_at_all_procs: bool = False

    # dataset
    evaluate_dataset: str = "sample_class"
    sample_class: SampleClassDataProviderConfig = field(default_factory=SampleClassDataProviderConfig)

    # autoencoder
    autoencoder: Optional[str] = None
    autoencoder_dtype: str = "fp32"
    scaling_factor: Optional[float] = None

    # model
    model: str = "uvit"
    dit: DiTConfig = field(default_factory=DiTConfig)
    uvit: UViTConfig = field(default_factory=UViTConfig)

    # metrics
    compute_fid: bool = True
    fid: FIDStatsConfig = field(default_factory=FIDStatsConfig)
    compute_inception_score: bool = True
    inception_score: InceptionScoreStatsConfig = field(default_factory=InceptionScoreStatsConfig)


class Evaluator:
    def __init__(self, cfg: EvaluatorConfig):
        self.cfg = cfg
        self.setup_dist_env()
        self.setup_seed()

        # data provider
        if cfg.evaluate_dataset == "sample_class":
            self.evaluate_data_provider = SampleClassDataProvider(cfg.sample_class)
        else:
            raise NotImplementedError

        # autoencoder
        if cfg.autoencoder is not None:
            device = torch.device("cuda")
            dtype = get_dtype_from_str(cfg.autoencoder_dtype)
            if cfg.autoencoder in REGISTERED_DCAE_MODEL:
                self.autoencoder = (
                    DCAE_HF.from_pretrained(f"mit-han-lab/{cfg.autoencoder}").eval().to(device=device, dtype=dtype)
                )
                assert cfg.scaling_factor is not None
            elif cfg.autoencoder in ["stabilityai/sd-vae-ft-ema", "flux-vae"]:
                self.autoencoder = AutoencoderKL(cfg.autoencoder).eval().to(device=device, dtype=dtype)
                cfg.scaling_factor = self.autoencoder.model.config.scaling_factor
            else:
                raise ValueError(f"{cfg.model} is not supported for evaluating and training")

        # model
        if cfg.model == "dit":
            cfg.dit.input_size = cfg.resolution // self.autoencoder.spatial_compression_ratio
            model = DiT(cfg.dit)
        elif cfg.model == "uvit":
            cfg.uvit.input_size = cfg.resolution // self.autoencoder.spatial_compression_ratio
            model = UViT(cfg.uvit)
        else:
            raise NotImplementedError

        if is_dist_initialized():
            self.model = nn.parallel.DistributedDataParallel(
                model.cuda(), device_ids=[get_dist_local_rank()], static_graph=True
            )
        else:
            self.model = model.cuda()
        self.rank = get_dist_rank()
        self.dist_size = get_dist_size()

    def setup_seed(self) -> None:
        seed = get_dist_rank() + self.cfg.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist_env(self) -> None:
        dist_init()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(get_dist_local_rank())

    @property
    def enable_amp(self) -> bool:
        return self.cfg.amp != "fp32"

    @property
    def amp_dtype(self) -> torch.dtype:
        return get_dtype_from_str(self.cfg.amp)

    @property
    def network(self) -> DiT | UViT:
        return self.model.module if is_parallel(self.model) else self.model

    @torch.no_grad()
    def evaluate(self, step: int, network: Optional[nn.Module] = None, f_log=sys.stdout) -> dict[str, Any]:
        if network is None:
            network = self.network
        network.eval()
        eval_generator = torch.Generator(device=torch.device("cuda"))
        eval_generator.manual_seed(self.cfg.seed + self.rank)

        if self.cfg.evaluate_split == "train":
            dataloader = self.evaluate_data_provider.train
        elif self.cfg.evaluate_split == "valid":
            dataloader = self.evaluate_data_provider.valid
        elif self.cfg.evaluate_split == "test":
            dataloader = self.evaluate_data_provider.test
        else:
            raise NotImplementedError

        # metrics
        if self.cfg.compute_fid:
            fid_stats = FIDStats(self.cfg.fid)
        if self.cfg.compute_inception_score:
            inception_score_stats = InceptionScoreStats(self.cfg.inception_score)

        if self.cfg.evaluate_dir_name is not None:
            evaluate_dir = os.path.join(self.cfg.run_dir, self.cfg.evaluate_dir_name)
        else:
            evaluate_dir = os.path.join(self.cfg.run_dir, f"{step}")
        if is_master():
            os.makedirs(evaluate_dir, exist_ok=True)
        if is_dist_initialized():
            dist_barrier()

        with tqdm(
            total=len(dataloader),
            desc="Valid Step #{}".format(step),
            disable=not is_master(),
            file=f_log,
            mininterval=10.0,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ) as t:
            num_saved_images = 0
            for _, (inputs, inputs_null) in enumerate(dataloader):
                # preprocessing
                inputs = inputs.cuda()
                inputs_null = inputs_null.cuda()
                # sample
                if self.cfg.model in ["dit", "uvit"]:
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
                        latent_samples = network.generate(inputs, inputs_null, self.cfg.cfg_scale, eval_generator)
                    latent_samples = (
                        latent_samples.to(dtype=get_dtype_from_str(self.cfg.autoencoder_dtype))
                        / self.cfg.scaling_factor
                    )
                    image_samples = self.autoencoder.decode(latent_samples)
                    assert torch.isnan(image_samples).sum() == 0, "NaN detected!"
                    image_samples_uint8 = torch.clamp(127.5 * image_samples + 128.0, 0, 255).to(dtype=torch.uint8)
                    image_samples_numpy = image_samples_uint8.permute(0, 2, 3, 1).cpu().numpy()
                else:
                    raise ValueError(f"diffusion model {self.cfg.model} is not supported")

                if (
                    num_saved_images < self.cfg.num_save_images and (is_master() or self.cfg.save_images_at_all_procs)
                ) or self.cfg.save_all_images:
                    image_samples_PIL = [Image.fromarray(image) for image in image_samples_numpy]
                    for j, image_sample_PIL in enumerate(image_samples_PIL):
                        if self.cfg.save_all_images:
                            idx = num_saved_images * self.dist_size + self.rank
                        else:
                            if num_saved_images >= self.cfg.num_save_images:
                                break
                            idx = num_saved_images
                        image_sample_PIL.save(
                            os.path.join(evaluate_dir, f"{self.rank}_{idx:05d}_{inputs[j].item()}.png")
                        )
                        num_saved_images += 1
                    del image_samples_PIL

                ## fid
                if self.cfg.compute_fid:
                    fid_stats.add_data(image_samples_uint8)
                if self.cfg.compute_inception_score:
                    inception_score_stats.add_data(image_samples_uint8)
                ## tqdm
                t.update()

        valid_info_dict = dict()
        torch.cuda.empty_cache()
        # fid
        if self.cfg.compute_fid:
            valid_info_dict["fid"] = fid_stats.compute_fid()
        # compute_inception_score
        if self.cfg.compute_inception_score:
            valid_info_dict.update(inception_score_stats.compute())
        return valid_info_dict
