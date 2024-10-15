import argparse
import os
import sys

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from efficientvit.apps.utils import export_onnx
from efficientvit.cls_model_zoo import create_efficientvit_cls_model
from efficientvit.models.utils import val2tuple
from efficientvit.seg_model_zoo import create_efficientvit_seg_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_path", type=str)
    parser.add_argument("--task", type=str, default="cls", choices=["cls", "seg"])
    parser.add_argument("--model", type=str, default="efficientvit-l2")
    parser.add_argument("--resolution", type=int, nargs="+", default=224)
    parser.add_argument("--bs", help="batch size", type=int, default=16)
    parser.add_argument("--op_set", type=int, default=11)

    args = parser.parse_args()

    resolution = val2tuple(args.resolution, 2)
    if args.task == "cls":
        model = create_efficientvit_cls_model(name=args.model, pretrained=False)
    elif args.task == "seg":
        model = create_efficientvit_seg_model(name=args.model, pretrained=False)
    else:
        raise NotImplementedError

    dummy_input = torch.rand((args.bs, 3, *resolution))
    export_onnx(model, args.export_path, dummy_input, simplify=True, opset=args.op_set)


if __name__ == "__main__":
    main()
