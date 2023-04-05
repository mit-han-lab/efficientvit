from typing import List, Dict

import torch
import torch.nn as nn

from models.utils import build_kwargs_from_config
from models.nn import OpSequential, ConvLayer, LinearLayer
from models.efficientvit.backbone import EfficientViTBackbone

__all__ = [
    "EfficientViTCls",
    "efficientvit_cls_b1",
    "efficientvit_cls_b2",
    "efficientvit_cls_b3",
]


class ClsHead(OpSequential):
    def __init__(self, in_channels: int, width_list: List[int], n_classes=1000, dropout_rate=0.0, norm="bn2d", act_func="hswish", fid="stage_final"):
        ops = [
            ConvLayer(in_channels, width_list[0], 1, norm=norm, act_func=act_func),
            nn.AdaptiveAvgPool2d(output_size=1),
            LinearLayer(width_list[0], width_list[1], False, norm="ln", act_func=act_func),
            LinearLayer(width_list[1], n_classes, True, dropout_rate, None, None),
        ]
        super().__init__(ops)

        self.fid = fid
    
    def forward(self, feed_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = feed_dict[self.fid]
        return OpSequential.forward(self, x)


class EfficientViTCls(nn.Module):
    def __init__(self, backbone: EfficientViTBackbone, head: ClsHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feed_dict = self.backbone(x)
        output = self.head(feed_dict)
        return output


def efficientvit_cls_b1(**kwargs) -> EfficientViTCls:
    from models.efficientvit.backbone import efficientvit_backbone_b1
    backbone = efficientvit_backbone_b1(**kwargs)

    head = ClsHead(
        in_channels=256,
        width_list=[1536, 1600],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


def efficientvit_cls_b2(**kwargs) -> EfficientViTCls:
    from models.efficientvit.backbone import efficientvit_backbone_b2
    backbone = efficientvit_backbone_b2(**kwargs)

    head = ClsHead(
        in_channels=384,
        width_list=[2304, 2560],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


def efficientvit_cls_b3(**kwargs) -> EfficientViTCls:
    from models.efficientvit.backbone import efficientvit_backbone_b3
    backbone = efficientvit_backbone_b3(**kwargs)

    head = ClsHead(
        in_channels=512,
        width_list=[2304, 2560],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model
