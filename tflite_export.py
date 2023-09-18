# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import argparse

import torch
from tinynn.converter import TFLiteConverter

from efficientvit.cls_model_zoo import create_cls_model
from efficientvit.models.nn.ops import UpSampleLayer
from efficientvit.models.utils import val2tuple
from efficientvit.seg_model_zoo import create_seg_model

parser = argparse.ArgumentParser()
parser.add_argument("--export_path", type=str)
parser.add_argument("--task", type=str, default="cls", choices=["cls", "seg"])
parser.add_argument("--dataset", type=str, default="none", choices=["ade20k", "cityscapes"])
parser.add_argument("--model", type=str, default="b3")
parser.add_argument("--resolution", type=int, nargs="+", default=224)

if __name__ == "__main__":
    args = parser.parse_args()

    resolution = val2tuple(args.resolution, 2)
    if args.task == "cls":
        model = create_cls_model(
            name=args.model,
            pretrained=False,
        )
    elif args.task == "seg":
        model = create_seg_model(
            name=args.model,
            dataset=args.dataset,
            pretrained=False,
        )
        # bicubic upsampling is not supported in TFLite
        # replace it with bilinear upsampling
        for m in model.modules():
            if isinstance(m, UpSampleLayer):
                m.mode = "bilinear"
    else:
        raise NotImplementedError

    model.cpu()
    model.eval()
    dummy_input = torch.rand((1, 3, *resolution))
    with torch.no_grad():
        converter = TFLiteConverter(model, dummy_input, tflite_path=args.export_path)
        converter.convert()
