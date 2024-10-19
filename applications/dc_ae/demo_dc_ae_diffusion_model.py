import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import MISSING, OmegaConf
from torchvision.utils import save_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.diffusion_model_zoo import DCAE_Diffusion_HF
from efficientvit.models.utils.network import get_dtype_from_str


@dataclass
class DemoDiffusionModelConfig:
    model: str = MISSING
    diffusion_model_dtype: str = "fp32"
    autoencoder_dtype: str = "fp32"
    cfg_scale: float = 6.0
    run_dir: str = MISSING


def main():
    torch.set_grad_enabled(False)
    cfg: DemoDiffusionModelConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(DemoDiffusionModelConfig), OmegaConf.from_cli())
    )

    device = torch.device("cuda")
    dc_ae_diffusion = DCAE_Diffusion_HF.from_pretrained(f"mit-han-lab/{cfg.model}")
    dc_ae_diffusion.autoencoder = dc_ae_diffusion.autoencoder.to(
        device=device, dtype=get_dtype_from_str(cfg.autoencoder_dtype)
    )
    dc_ae_diffusion.diffusion_model = dc_ae_diffusion.diffusion_model.to(
        device=device, dtype=get_dtype_from_str(cfg.diffusion_model_dtype)
    )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    eval_generator = torch.Generator(device=device)
    eval_generator.manual_seed(0)
    inputs = torch.tensor(
        [279, 333, 979, 936, 933, 145, 497, 1, 248, 360, 793, 12, 387, 437, 938, 978], dtype=torch.int, device=device
    )
    num_samples = inputs.shape[0]
    inputs_null = 1000 * torch.ones((num_samples,), dtype=torch.int, device=device)
    latent_samples = dc_ae_diffusion.diffusion_model.generate(inputs, inputs_null, cfg.cfg_scale, eval_generator)
    latent_samples = latent_samples.to(dtype=get_dtype_from_str(cfg.autoencoder_dtype)) / dc_ae_diffusion.scaling_factor
    image_samples = dc_ae_diffusion.autoencoder.decode(latent_samples)
    save_path = os.path.join(cfg.run_dir, "demo.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"saving demo image to {save_path}")
    save_image(image_samples * 0.5 + 0.5, save_path, nrow=int(np.sqrt(num_samples)))


if __name__ == "__main__":
    main()
