from copy import deepcopy

from efficientvit.gazesamcore.evit.helpers import EvitResize

__all__ = ["run_evit_model_pytorch", "run_evit_model_onnx", "run_evit_model_tensorrt"]


def run_evit_model_pytorch(img, bboxes, model, timer=None):
    boxes = deepcopy(bboxes)
    if timer is not None:
        timer.start("evit encoder")
    model.set_image(img)
    if timer is not None:
        timer.stop("evit encoder")

    masks, iou_preds = model.predict(boxes=boxes, multimask_output=True, return_logits=False, timer=timer)
    return masks, iou_preds


def run_evit_model_onnx(img, bboxes, model, timer=None):
    origin_image_size = img.shape[:2]
    boxes = deepcopy(bboxes)

    model.set_image(img, timer=timer)
    masks, iou_preds = model.predict_torch(im_size=origin_image_size, boxes=boxes, timer=timer)
    return masks, iou_preds


def run_evit_model_tensorrt(img, bboxes, model):
    boxes = deepcopy(bboxes)

    origin_image_size = img.shape[:2]
    input_size = EvitResize.get_preprocess_shape(*origin_image_size, long_side_length=1024)

    model.set_image(img)
    masks, iou_preds = model.predict_torch(im_size=input_size, boxes=boxes)

    return masks, iou_preds
