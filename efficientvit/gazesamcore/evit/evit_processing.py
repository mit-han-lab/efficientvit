from efficientvit.gazesamcore.evit.runtime_variants import (
    run_evit_model_onnx,
    run_evit_model_pytorch,
    run_evit_model_tensorrt,
)
from efficientvit.gazesamcore.utils import ONNX, PYTORCH, TENSORRT

__all__ = ["get_evit_masks"]


def get_evit_masks(img, bboxes, model, mode, timer=None):
    if timer is not None:
        timer.start("evit")

    if mode == PYTORCH:
        masks, iou_preds = run_evit_model_pytorch(img, bboxes, model, timer=timer)

    elif mode == ONNX:
        masks, iou_preds = run_evit_model_onnx(img, bboxes, model, timer=timer)

    elif mode == TENSORRT:
        masks, iou_preds = run_evit_model_tensorrt(img, bboxes, model)

    else:
        raise NotImplementedError(f"{mode} mode not implemented for evit image segmentation")

    if timer is not None:
        timer.stop("evit")

    return masks, iou_preds
