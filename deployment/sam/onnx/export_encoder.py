import argparse
import os
import sys
import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
sys.path.append(ROOT_DIR)

from efficientvit.models.efficientvit.sam import EfficientViTSam
from efficientvit.sam_model_zoo import create_sam_model


class SamResize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            return self.apply_image(image)
        else:
            return image.permute(2, 0, 1)

    def apply_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects a torch tensor with shape HxWxC in float format.
        """

        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.size)
        return resize(image.permute(2, 0, 1), target_size)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"


class EncoderOnnxModel(nn.Module):
    def __init__(
        self,
        model: EfficientViTSam,
    ):
        super().__init__()

        self.model = model
        self.image_size = self.model.image_size
        self.image_encoder = self.model.image_encoder
        self.transform = SamResize(size=self.image_size[1])

    @torch.no_grad()
    def forward(self, input_image):
        image_embeddings = self.image_encoder(input_image)
        return image_embeddings


def run_export(
    model: str,
    weight_url: str,
    output: str,
    opset: int,
) -> None:
    print("Loading model...")
    efficientvit_sam = create_sam_model(model, True, weight_url).eval()

    onnx_model = EncoderOnnxModel(model=efficientvit_sam)

    if model in ["l0", "l1", "l2"]:
        image_size = [512, 512]
    elif model in ["xl0", "xl1"]:
        image_size = [1024, 1024]
    else:
        raise NotImplementedError

    dummy_input = {"input_image": torch.randn((1, 3, image_size[0], image_size[1]), dtype=torch.float)}
    dynamic_axes = {
        "input_image": {0: "batch_size"},
    }

    _ = onnx_model(**dummy_input)

    output_names = ["image_embeddings"]

    if not os.path.exists(os.path.dirname(output)):
        os.makedirs(os.path.dirname(output))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        print(f"Exporting onnx model to {output}...")
        with open(output, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_input.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_input.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--weight_url", type=str)
    parser.add_argument("--output", type=str, required=True, help="The filename to save the onnx model to.")
    parser.add_argument("--opset", type=int, default=17, help="The ONNX opset version to use. Must be >=11.")
    args = parser.parse_args()

    run_export(
        model=args.model,
        weight_url=args.weight_url,
        output=args.output,
        opset=args.opset,
    )
