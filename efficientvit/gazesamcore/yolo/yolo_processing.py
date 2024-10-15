import numpy as np

from efficientvit.gazesamcore.utils import ONNX, PYTORCH, TENSORRT
from efficientvit.gazesamcore.yolo.runtime_variants import run_yolo_model_onnx, run_yolo_model_tensorrt

__all__ = ["get_yolo_bboxes", "get_bboxes"]


def get_yolo_bboxes(frame, model, mode, timer=None):
    if timer is not None:
        timer.start("yolo")

    if mode == PYTORCH:
        preds = model.predict(frame)
        pred_bboxes = preds.prediction.bboxes_xyxy.astype(int)
        mask = ~(pred_bboxes[:, 0] == pred_bboxes[:, 2]) & ~(pred_bboxes[:, 1] == pred_bboxes[:, 3])
        bboxes = pred_bboxes[mask]

    elif mode == ONNX:
        pred_bboxes = run_yolo_model_onnx(frame, model)
        bboxes = get_bboxes(frame, pred_bboxes)

    elif mode == TENSORRT:
        pred_bboxes = run_yolo_model_tensorrt(frame, model)
        bboxes = get_bboxes(frame, pred_bboxes)

    else:
        raise NotImplementedError(f"{mode} mode not implemented for YOLO object detection")

    if timer is not None:
        timer.stop("yolo")

    return bboxes


def get_bboxes(img, pred_bboxes):
    h, w, _ = img.shape
    yolo_side_len = 640.0
    w_sf, h_sf = w / yolo_side_len, h / yolo_side_len
    scales = np.array([w_sf, h_sf, w_sf, h_sf])

    x1, x2 = pred_bboxes[:, 0], pred_bboxes[:, 2]
    y1, y2 = pred_bboxes[:, 1], pred_bboxes[:, 3]
    boxes = np.column_stack((x1, y1, x2, y2))

    mask = ~(boxes[:, 0] == boxes[:, 2]) & ~(boxes[:, 1] == boxes[:, 3])
    nondup_boxes = boxes[mask]

    scaled_boxes = (nondup_boxes * scales).astype(np.int32)
    return scaled_boxes
