from typing import Callable, Optional

from huggingface_hub import PyTorchModelHubMixin

from efficientvit.models.efficientvit.dc_ae import DCAE, DCAEConfig, dc_ae_f32c32, dc_ae_f64c128, dc_ae_f128c512

__all__ = ["create_dc_ae_model_cfg", "DCAE_HF"]


REGISTERED_DCAE_MODEL: dict[str, tuple[Callable, Optional[str]]] = {
    "dc-ae-f32c32-in-1.0": (dc_ae_f32c32, None),
    "dc-ae-f64c128-in-1.0": (dc_ae_f64c128, None),
    "dc-ae-f128c512-in-1.0": (dc_ae_f128c512, None),
    #################################################################################################
    "dc-ae-f32c32-mix-1.0": (dc_ae_f32c32, None),
    "dc-ae-f64c128-mix-1.0": (dc_ae_f64c128, None),
    "dc-ae-f128c512-mix-1.0": (dc_ae_f128c512, None),
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
