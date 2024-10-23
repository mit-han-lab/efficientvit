from typing import Callable, Optional

import diffusers
import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from efficientvit.models.efficientvit.dc_ae import DCAE, DCAEConfig, dc_ae_f32c32, dc_ae_f64c128, dc_ae_f128c512

__all__ = ["create_dc_ae_model_cfg", "DCAE_HF", "AutoencoderKL"]


REGISTERED_DCAE_MODEL: dict[str, tuple[Callable, Optional[str]]] = {
    "dc-ae-f32c32-in-1.0": (dc_ae_f32c32, None),
    "dc-ae-f64c128-in-1.0": (dc_ae_f64c128, None),
    "dc-ae-f128c512-in-1.0": (dc_ae_f128c512, None),
    #################################################################################################
    "dc-ae-f32c32-mix-1.0": (dc_ae_f32c32, None),
    "dc-ae-f64c128-mix-1.0": (dc_ae_f64c128, None),
    "dc-ae-f128c512-mix-1.0": (dc_ae_f128c512, None),
    #################################################################################################
    "dc-ae-f32c32-sana-1.0": (dc_ae_f32c32, None),
}


def create_dc_ae_model_cfg(name: str, pretrained_path: Optional[str] = None) -> DCAEConfig:
    assert name in REGISTERED_DCAE_MODEL, f"{name} is not supported"
    dc_ae_cls, default_pt_path = REGISTERED_DCAE_MODEL[name]
    pretrained_path = default_pt_path if pretrained_path is None else pretrained_path
    model_cfg = dc_ae_cls(name, pretrained_path)
    return model_cfg


class DCAE_HF(DCAE, PyTorchModelHubMixin):
    def __init__(self, model_name: str):
        cfg = create_dc_ae_model_cfg(model_name)
        DCAE.__init__(self, cfg)


class AutoencoderKL(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        if self.model_name in ["stabilityai/sd-vae-ft-ema"]:
            self.model = diffusers.models.AutoencoderKL.from_pretrained(self.model_name)
            self.spatial_compression_ratio = 8
        elif self.model_name == "flux-vae":
            from diffusers import FluxPipeline

            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
            self.model = diffusers.models.AutoencoderKL.from_pretrained(pipe.vae.config._name_or_path)
            self.spatial_compression_ratio = 8
        else:
            raise ValueError(f"{self.model_name} is not supported for AutoencoderKL")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_name in ["stabilityai/sd-vae-ft-ema", "flux-vae"]:
            return self.model.encode(x).latent_dist.sample()
        else:
            raise ValueError(f"{self.model_name} is not supported for AutoencoderKL")

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.model_name in ["stabilityai/sd-vae-ft-ema", "flux-vae"]:
            return self.model.decode(latent).sample
        else:
            raise ValueError(f"{self.model_name} is not supported for AutoencoderKL")
