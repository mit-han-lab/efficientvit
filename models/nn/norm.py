from typing import Dict, Optional, Type

import torch.nn as nn

from models.utils import build_kwargs_from_config

__all__ = ["build_norm"]


# register normalization function here
REGISTERED_NORM_DICT: Dict[str, Type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
}


def build_norm(name="bn2d", num_features=None, **kwargs) -> Optional[nn.Module]:
    if name == "ln":
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None
