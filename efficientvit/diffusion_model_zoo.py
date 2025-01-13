from typing import Callable, Optional

import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

from efficientvit.diffusioncore.evaluator import Evaluator, EvaluatorConfig
from efficientvit.diffusioncore.models.dit import dc_ae_dit_xl_in_512px
from efficientvit.diffusioncore.models.uvit import (
    dc_ae_usit_2b_in_512px,
    dc_ae_usit_h_in_512px,
    dc_ae_uvit_2b_in_512px,
    dc_ae_uvit_h_in_512px,
    dc_ae_uvit_s_in_512px,
)

__all__ = ["create_dc_ae_diffusion_model", "DCAE_Diffusion_HF"]


REGISTERED_DCAE_DIFFUSION_MODEL: dict[str, tuple[Callable, str, float, int, Optional[str]]] = {
    "dc-ae-f32c32-in-1.0-dit-xl-in-512px": (
        dc_ae_dit_xl_in_512px,
        "dc-ae-f32c32-in-1.0",
        0.3189,
        32,
        None,
    ),
    "dc-ae-f32c32-in-1.0-dit-xl-in-512px-trainbs1024": (
        dc_ae_dit_xl_in_512px,
        "dc-ae-f32c32-in-1.0",
        0.3189,
        32,
        None,
    ),
    ################################################################################
    "dc-ae-f32c32-in-1.0-uvit-s-in-512px": (
        dc_ae_uvit_s_in_512px,
        "dc-ae-f32c32-in-1.0",
        0.3189,
        32,
        None,
    ),
    "dc-ae-f32c32-in-1.0-uvit-h-in-512px": (
        dc_ae_uvit_h_in_512px,
        "dc-ae-f32c32-in-1.0",
        0.3189,
        32,
        None,
    ),
    "dc-ae-f32c32-in-1.0-uvit-2b-in-512px": (
        dc_ae_uvit_2b_in_512px,
        "dc-ae-f32c32-in-1.0",
        0.3189,
        32,
        None,
    ),
    ################################################################################
    "dc-ae-f32c32-in-1.0-usit-h-in-512px": (
        dc_ae_usit_h_in_512px,
        "dc-ae-f32c32-in-1.0",
        0.3189,
        32,
        None,
    ),
    "dc-ae-f32c32-in-1.0-usit-2b-in-512px": (
        dc_ae_usit_2b_in_512px,
        "dc-ae-f32c32-in-1.0",
        0.3189,
        32,
        None,
    ),
    ################################################################################
    "dc-ae-f64c128-in-1.0-uvit-h-in-512px": (
        dc_ae_uvit_h_in_512px,
        "dc-ae-f64c128-in-1.0",
        0.2889,
        128,
        None,
    ),
    "dc-ae-f64c128-in-1.0-uvit-h-in-512px-train2000k": (
        dc_ae_uvit_h_in_512px,
        "dc-ae-f64c128-in-1.0",
        0.2889,
        128,
        None,
    ),
    "dc-ae-f64c128-in-1.0-uvit-2b-in-512px": (
        dc_ae_uvit_2b_in_512px,
        "dc-ae-f64c128-in-1.0",
        0.2889,
        128,
        None,
    ),
    "dc-ae-f64c128-in-1.0-uvit-2b-in-512px-train2000k": (
        dc_ae_uvit_2b_in_512px,
        "dc-ae-f64c128-in-1.0",
        0.2889,
        128,
        None,
    ),
}


def create_dc_ae_diffusion_model_cfg(name: str, pretrained_path: Optional[str] = None) -> EvaluatorConfig:
    diffusion_cls, ae_name, scaling_factor, in_channels, default_pt = REGISTERED_DCAE_DIFFUSION_MODEL[name]
    pretrained_path = default_pt if pretrained_path is None else pretrained_path
    cfg_str = diffusion_cls(ae_name, scaling_factor, in_channels, pretrained_path)
    cfg = OmegaConf.from_dotlist(cfg_str.split(" ") + ["run_dir=tmp"])
    cfg: EvaluatorConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(EvaluatorConfig), cfg))
    return cfg


class DCAE_Diffusion_HF(nn.Module, PyTorchModelHubMixin):
    def __init__(self, model_name: str):
        super().__init__()
        cfg = create_dc_ae_diffusion_model_cfg(model_name)
        evaluator = Evaluator(cfg)
        self.autoencoder, self.diffusion_model, self.scaling_factor = (
            evaluator.autoencoder,
            evaluator.network,
            cfg.scaling_factor,
        )
