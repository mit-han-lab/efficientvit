from efficientvit.models.efficientvit import (
    EfficientViTCls,
    efficientvit_cls_b0,
    efficientvit_cls_b1,
    efficientvit_cls_b2,
    efficientvit_cls_b3,
)
from efficientvit.models.utils import load_state_dict_from_file

__all__ = ["create_cls_model"]


REGISTERED_CLS_MODEL: dict[str, str] = {
    "b0-r224": "assets/checkpoints/cls/b0-r224.pt",
    ###############################################################################
    "b1-r224": "assets/checkpoints/cls/b1-r224.pt",
    "b1-r256": "assets/checkpoints/cls/b1-r256.pt",
    "b1-r288": "assets/checkpoints/cls/b1-r288.pt",
    ###############################################################################
    "b2-r224": "assets/checkpoints/cls/b2-r224.pt",
    "b2-r256": "assets/checkpoints/cls/b2-r256.pt",
    "b2-r288": "assets/checkpoints/cls/b2-r288.pt",
    ###############################################################################
    "b3-r224": "assets/checkpoints/cls/b3-r224.pt",
    "b3-r256": "assets/checkpoints/cls/b3-r256.pt",
    "b3-r288": "assets/checkpoints/cls/b3-r288.pt",
    ###############################################################################
}


def create_cls_model(name: str, pretrained=True, weight_url: str or None = None, **kwargs) -> EfficientViTCls:
    model_dict = {
        "b0": efficientvit_cls_b0,
        "b1": efficientvit_cls_b1,
        "b2": efficientvit_cls_b2,
        "b3": efficientvit_cls_b3,
    }

    model_id = name.split("-")[0]
    if model_id not in model_dict:
        raise ValueError(f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}")
    else:
        model = model_dict[model_id](**kwargs)

    if pretrained:
        weight_url = weight_url or REGISTERED_CLS_MODEL.get(name, None)
        if weight_url is None:
            raise ValueError(f"Do not find the pretrained weight of {name}.")
        else:
            weight = load_state_dict_from_file(weight_url)
            model.load_state_dict(weight)
    return model
