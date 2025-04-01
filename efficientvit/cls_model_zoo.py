from typing import Callable, Optional

from efficientvit.models.efficientvit import (
    EfficientViTCls,
    efficientvit_cls_b0,
    efficientvit_cls_b1,
    efficientvit_cls_b2,
    efficientvit_cls_b3,
    efficientvit_cls_l1,
    efficientvit_cls_l2,
    efficientvit_cls_l3,
)
from efficientvit.models.nn.norm import set_norm_eps
from efficientvit.models.utils import load_state_dict_from_file

__all__ = ["create_efficientvit_cls_model"]


REGISTERED_EFFICIENTVIT_CLS_MODEL: dict[str, tuple[Callable, float, Optional[str]]] = {
    "efficientvit-b0": (efficientvit_cls_b0, 1e-5, None),
    "efficientvit-b0-r224": (efficientvit_cls_b0, 1e-5, "assets/checkpoints/efficientvit_cls/efficientvit_b0_r224.pt"),
    ##################################################################################################################
    "efficientvit-b1": (efficientvit_cls_b1, 1e-5, None),
    "efficientvit-b1-r224": (efficientvit_cls_b1, 1e-5, "assets/checkpoints/efficientvit_cls/efficientvit_b1_r224.pt"),
    "efficientvit-b1-r256": (efficientvit_cls_b1, 1e-5, "assets/checkpoints/efficientvit_cls/efficientvit_b1_r256.pt"),
    "efficientvit-b1-r288": (efficientvit_cls_b1, 1e-5, "assets/checkpoints/efficientvit_cls/efficientvit_b1_r288.pt"),
    ##################################################################################################################
    "efficientvit-b2": (efficientvit_cls_b2, 1e-5, None),
    "efficientvit-b2-r224": (efficientvit_cls_b2, 1e-5, "assets/checkpoints/efficientvit_cls/efficientvit_b2_r224.pt"),
    "efficientvit-b2-r256": (efficientvit_cls_b2, 1e-5, "assets/checkpoints/efficientvit_cls/efficientvit_b2_r256.pt"),
    "efficientvit-b2-r288": (efficientvit_cls_b2, 1e-5, "assets/checkpoints/efficientvit_cls/efficientvit_b2_r288.pt"),
    ##################################################################################################################
    "efficientvit-b3": (efficientvit_cls_b3, 1e-5, None),
    "efficientvit-b3-r224": (efficientvit_cls_b3, 1e-5, "assets/checkpoints/efficientvit_cls/efficientvit_b3_r224.pt"),
    "efficientvit-b3-r256": (efficientvit_cls_b3, 1e-5, "assets/checkpoints/efficientvit_cls/efficientvit_b3_r256.pt"),
    "efficientvit-b3-r288": (efficientvit_cls_b3, 1e-5, "assets/checkpoints/efficientvit_cls/efficientvit_b3_r288.pt"),
    ##################################################################################################################
    "efficientvit-l1": (efficientvit_cls_l1, 1e-7, None),
    "efficientvit-l1-r224": (efficientvit_cls_l1, 1e-7, "assets/checkpoints/efficientvit_cls/efficientvit_l1_r224.pt"),
    ##################################################################################################################
    "efficientvit-l2": (efficientvit_cls_l2, 1e-7, None),
    "efficientvit-l2-r224": (efficientvit_cls_l2, 1e-7, "assets/checkpoints/efficientvit_cls/efficientvit_l2_r224.pt"),
    "efficientvit-l2-r256": (efficientvit_cls_l2, 1e-7, "assets/checkpoints/efficientvit_cls/efficientvit_l2_r256.pt"),
    "efficientvit-l2-r288": (efficientvit_cls_l2, 1e-7, "assets/checkpoints/efficientvit_cls/efficientvit_l2_r288.pt"),
    "efficientvit-l2-r320": (efficientvit_cls_l2, 1e-7, "assets/checkpoints/efficientvit_cls/efficientvit_l2_r320.pt"),
    "efficientvit-l2-r384": (efficientvit_cls_l2, 1e-7, "assets/checkpoints/efficientvit_cls/efficientvit_l2_r384.pt"),
    ##################################################################################################################
    "efficientvit-l3": (efficientvit_cls_l3, 1e-7, None),
    "efficientvit-l3-r224": (efficientvit_cls_l3, 1e-7, "assets/checkpoints/efficientvit_cls/efficientvit_l3_r224.pt"),
    "efficientvit-l3-r256": (efficientvit_cls_l3, 1e-7, "assets/checkpoints/efficientvit_cls/efficientvit_l3_r256.pt"),
    "efficientvit-l3-r288": (efficientvit_cls_l3, 1e-7, "assets/checkpoints/efficientvit_cls/efficientvit_l3_r288.pt"),
    "efficientvit-l3-r320": (efficientvit_cls_l3, 1e-7, "assets/checkpoints/efficientvit_cls/efficientvit_l3_r320.pt"),
    "efficientvit-l3-r384": (efficientvit_cls_l3, 1e-7, "assets/checkpoints/efficientvit_cls/efficientvit_l3_r384.pt"),
}


def create_efficientvit_cls_model(
    name: str, pretrained=True, weight_url: Optional[str] = None, **kwargs
) -> EfficientViTCls:
    if name not in REGISTERED_EFFICIENTVIT_CLS_MODEL:
        raise ValueError(
            f"Cannot find {name} in the model zoo. List of models: {list(REGISTERED_EFFICIENTVIT_CLS_MODEL.keys())}"
        )
    else:
        model_cls, norm_eps, default_pt = REGISTERED_EFFICIENTVIT_CLS_MODEL[name]
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
