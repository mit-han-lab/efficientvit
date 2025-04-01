from typing import Callable, Optional

from efficientvit.models.efficientvit import (
    EfficientViTSam,
    efficientvit_sam_l0,
    efficientvit_sam_l1,
    efficientvit_sam_l2,
    efficientvit_sam_xl0,
    efficientvit_sam_xl1,
)
from efficientvit.models.nn.norm import set_norm_eps
from efficientvit.models.utils import load_state_dict_from_file

__all__ = ["create_efficientvit_sam_model"]


REGISTERED_EFFICIENTVIT_SAM_MODEL: dict[str, tuple[Callable, float, str]] = {
    "efficientvit-sam-l0": (efficientvit_sam_l0, 1e-6, "assets/checkpoints/efficientvit_sam/efficientvit_sam_l0.pt"),
    "efficientvit-sam-l1": (efficientvit_sam_l1, 1e-6, "assets/checkpoints/efficientvit_sam/efficientvit_sam_l1.pt"),
    "efficientvit-sam-l2": (efficientvit_sam_l2, 1e-6, "assets/checkpoints/efficientvit_sam/efficientvit_sam_l2.pt"),
    "efficientvit-sam-xl0": (efficientvit_sam_xl0, 1e-6, "assets/checkpoints/efficientvit_sam/efficientvit_sam_xl0.pt"),
    "efficientvit-sam-xl1": (efficientvit_sam_xl1, 1e-6, "assets/checkpoints/efficientvit_sam/efficientvit_sam_xl1.pt"),
}


def create_efficientvit_sam_model(
    name: str, pretrained=True, weight_url: Optional[str] = None, **kwargs
) -> EfficientViTSam:
    if name not in REGISTERED_EFFICIENTVIT_SAM_MODEL:
        raise ValueError(
            f"Cannot find {name} in the model zoo. List of models: {list(REGISTERED_EFFICIENTVIT_SAM_MODEL.keys())}"
        )
    else:
        model_cls, norm_eps, default_pt = REGISTERED_EFFICIENTVIT_SAM_MODEL[name]
        model = model_cls(**kwargs)
        set_norm_eps(model, norm_eps)
        weight_url = default_pt if weight_url is None else weight_url

    if pretrained:
        if weight_url is None:
            raise ValueError(f"Cannot find the pretrained weight of {name}.")
        else:
            weight = load_state_dict_from_file(weight_url)
            model.load_state_dict(weight)
    return model
