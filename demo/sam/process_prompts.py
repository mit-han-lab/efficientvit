from demo.sam.helpers.segmentation.process_prompts_onnx import (
    segment_full_img_onnx,
    segment_using_boxes_onnx,
    segment_using_points_and_boxes_onnx,
    segment_using_points_onnx,
)
from demo.sam.helpers.segmentation.process_prompts_pytorch import (
    segment_full_img_pytorch,
    segment_using_boxes_pytorch,
    segment_using_points_and_boxes_pytorch,
    segment_using_points_pytorch,
)
from demo.sam.helpers.segmentation.process_prompts_tensorrt import (
    segment_full_img_tensorrt,
    segment_using_boxes_tensorrt,
    segment_using_points_and_boxes_tensorrt,
    segment_using_points_tensorrt,
)
from demo.sam.helpers.utils import ONNX, PYTORCH, TENSORRT


def process_points(*args, runtime):
    if runtime == PYTORCH:
        return segment_using_points_pytorch(*args)
    elif runtime == ONNX:
        return segment_using_points_onnx(*args)
    elif runtime == TENSORRT:
        return segment_using_points_tensorrt(*args)
    else:
        raise NotImplementedError


def process_boxes(*args, runtime):
    if runtime == PYTORCH:
        return segment_using_boxes_pytorch(*args)
    elif runtime == ONNX:
        return segment_using_boxes_onnx(*args)
    elif runtime == TENSORRT:
        return segment_using_boxes_tensorrt(*args)
    else:
        raise NotImplementedError


def process_points_and_boxes(*args, runtime):
    if runtime == PYTORCH:
        return segment_using_points_and_boxes_pytorch(*args)
    elif runtime == ONNX:
        return segment_using_points_and_boxes_onnx(*args)
    elif runtime == TENSORRT:
        return segment_using_points_and_boxes_tensorrt(*args)
    else:
        raise NotImplementedError


def process_full_img(*args, runtime):
    if runtime == PYTORCH:
        return segment_full_img_pytorch(*args)
    elif runtime == ONNX:
        return segment_full_img_onnx(*args)
    elif runtime == TENSORRT:
        return segment_full_img_tensorrt(*args)
    else:
        raise NotImplementedError
