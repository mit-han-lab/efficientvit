import argparse
import os
import pathlib
import sys
import warnings

import onnxruntime  # type: ignore
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))))
sys.path.append(ROOT_DIR)

from decoder_onnx_model import EvitSAMBoxDecoder

from efficientvit.sam_model_zoo import create_efficientvit_sam_model


def run_export(
    model_type: str,
    checkpoint: str,
    output: str,
    opset: int,
    return_single_mask: bool,
):
    print("Loading model...")
    efficientvit_sam = create_efficientvit_sam_model(model_type, True, checkpoint).eval()

    onnx_model = EvitSAMBoxDecoder(
        model=efficientvit_sam,
        return_single_mask=return_single_mask,
    )

    dynamic_axes = {
        "boxes": {0: "batch_size"},
    }

    embed_dim = efficientvit_sam.prompt_encoder.embed_dim
    embed_size = efficientvit_sam.prompt_encoder.image_embedding_size
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "boxes": torch.randint(low=0, high=1024, size=(3, 1, 4), dtype=torch.float),
    }

    _ = onnx_model(**dummy_inputs)

    output_names = ["low_res_masks", "iou_predictions"]

    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(output, "wb") as f:
            print(f"Exporting onnx model to {output}...")
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

    ort_inputs = {k: to_numpy(v) for k, v in dummy_inputs.items()}
    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(output, providers=providers)
    _ = ort_session.run(None, ort_inputs)


def to_numpy(tensor):
    return tensor.cpu().numpy()


parser = argparse.ArgumentParser(description="Export EfficientViT-SAM decoder to ONNX.")
parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    choices=[
        "efficientvit-sam-l0",
        "efficientvit-sam-l1",
        "efficientvit-sam-l2",
        "efficientvit-sam-xl0",
        "efficientvit-sam-xl1",
    ],
    help="EfficientViT SAM model to export",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="The path to the model checkpoint",
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Resulting ONNX's filename",
)
parser.add_argument(
    "--opset",
    type=int,
    default=17,
    help="ONNX opset version",
)

parser.add_argument(
    "--return-single-mask",
    action="store_true",
    help=(
        "If true, the exported ONNX model will only return the best mask, "
        "instead of returning multiple masks. For high resolution images "
        "this can improve runtime when upscaling masks is expensive."
    ),
)

if __name__ == "__main__":
    args = parser.parse_args()
    run_export(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        output=args.output,
        opset=args.opset,
        return_single_mask=args.return_single_mask,
    )
