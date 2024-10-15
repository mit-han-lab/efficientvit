import time

import numpy as np

from efficientvit.gazesamcore.face.helpers import demo_postprocess, multiclass_nms, yolox_preprocess
from efficientvit.gazesamcore.face.runtime_variants import (
    run_face_detection_model_onnx,
    run_face_detection_model_pytorch,
    run_face_detection_model_trt,
)
from efficientvit.gazesamcore.utils import ONNX, PYTORCH, TENSORRT

__all__ = ["detect_face", "get_face_bbox"]


def detect_face(img, model, mode, timer=None, score_thr=0.5, input_shape=(160, 128)):
    img, ratio = yolox_preprocess(img, input_shape)
    img = img[None, :, :, :]

    if timer is not None:
        timer.start("face detection")

    if mode == PYTORCH:
        output = run_face_detection_model_pytorch(img, model)
    elif mode == ONNX:
        output = run_face_detection_model_onnx(img, model)
    elif mode == TENSORRT:
        output = run_face_detection_model_trt(img, model)
    else:
        raise NotImplementedError(f"{mode} mode not implemented for face detection")

    if timer is not None:
        timer.stop("face detection")

    predictions = demo_postprocess(output, input_shape)[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio

    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=score_thr)
    if dets is not None:
        final_boxes, final_scores = dets[:, :4], dets[:, 4]
        return np.array([[*final_box, final_score] for final_box, final_score in zip(final_boxes, final_scores)])
    else:
        return None


def get_face_bbox(faces, bbox_smoother):
    if faces is None:
        return None

    current_timestamp = time.time()
    face = faces[0]

    x1, y1, x2, y2 = face[:4]
    [[x1, y1], [x2, y2]] = bbox_smoother([[x1, y1], [x2, y2]], t=current_timestamp)

    face = np.array([x1, y1, x2, y2, face[-1]])
    face_bbox = face[:4].astype(int)
    return face_bbox
