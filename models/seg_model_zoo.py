from typing import Optional, Dict

from models.utils import load_state_dict_from_file
from models.efficientvit import EfficientViTSeg

__all__ = ["create_seg_model"]


REGISTERED_SEG_MODEL: Dict[str, Dict[str, str]] = {
    "cityscapes": {
        "b0-r960": "assets/checkpoints/seg/cityscapes/b0-r960.pt",
        "b1-r896": "assets/checkpoints/seg/cityscapes/b1-r896.pt",
        "b2-r1024": "assets/checkpoints/seg/cityscapes/b2-r1024.pt",
        "b3-r1184": "assets/checkpoints/seg/cityscapes/b3-r1184.pt",
    },
    "ade20k": {
        "b1-r480": "assets/checkpoints/seg/ade20k/b1-r480.pt",
        "b2-r416": "assets/checkpoints/seg/ade20k/b2-r416.pt",
        "b3-r512": "assets/checkpoints/seg/ade20k/b3-r512.pt",
    }
}


def create_seg_model(name: str, dataset: str, pretrained=True, weight_url: Optional[str] = None, **kwargs) -> EfficientViTSeg:
    from models.efficientvit import efficientvit_seg_b0, efficientvit_seg_b1, efficientvit_seg_b2, efficientvit_seg_b3
    model_dict = {
        "b0": efficientvit_seg_b0,
        "b1": efficientvit_seg_b1,
        "b2": efficientvit_seg_b2,
        "b3": efficientvit_seg_b3,
    }

    model_id = name.split("-")[0]
    if model_id not in model_dict:
        raise ValueError(f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}")
    else:
        model = model_dict[model_id](dataset=dataset, **kwargs)
    
    if pretrained:
        weight_url = weight_url or REGISTERED_SEG_MODEL[dataset].get(name, None)
        if weight_url is None:
            raise ValueError(f"Do not find the pretrained weight of {name}.")
        else:
            weight = load_state_dict_from_file(weight_url)
            model.load_state_dict(weight)
    return model
