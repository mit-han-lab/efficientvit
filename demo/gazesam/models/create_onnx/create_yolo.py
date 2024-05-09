import argparse

from super_gradients.common.object_names import Models
from super_gradients.conversion import ExportTargetBackend
from super_gradients.training import models

parser = argparse.ArgumentParser(description="Create YOLO ONNX model")
parser.add_argument(
    "--model-size",
    type=str,
    default="m",
    choices=["s", "m", "l"],
    help="Model size (s, m, l)",
)
parser.add_argument(
    "--runtime",
    type=str,
    default="trt",
    choices=["trt", "onnx"],
    help="Export mode (trt, onnx)",
)
args = parser.parse_args()

if args.model_size == "s":
    model = Models.YOLO_NAS_S
elif args.model_size == "m":
    model = Models.YOLO_NAS_M
elif args.model_size == "l":
    model = Models.YOLO_NAS_L
else:
    raise ValueError("Invalid model size")

onnx_model = models.get(model, pretrained_weights="coco")

# need different version because the trt version includes an NMS TRT plugin node
if args.runtime == "trt":
    onnx_model.export(f"onnx/yolo_{args.model_size}.onnx", engine=ExportTargetBackend.TENSORRT)
elif args.runtime == "onnx":
    onnx_model.export(f"onnx/yolo_{args.model_size}_ort.onnx")
