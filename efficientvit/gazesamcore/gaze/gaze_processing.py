from efficientvit.gazesamcore.gaze.helpers import find_edge_intersection, get_point_along_gaze
from efficientvit.gazesamcore.gaze.runtime_variants import (
    run_gaze_estimation_model_onnx,
    run_gaze_estimation_model_tensorrt,
)
from efficientvit.gazesamcore.utils import ONNX, PYTORCH, TENSORRT

__all__ = ["estimate_gaze", "get_gaze_endpoints"]


def estimate_gaze(img, face_bbox, model, mode, timer=None):
    if timer is not None:
        timer.start("gaze estimation")

    if mode == PYTORCH:
        yawpitch = model(img)

    elif mode == ONNX:
        yawpitch = run_gaze_estimation_model_onnx(img, face_bbox, model, timer)

    elif mode == TENSORRT:
        yawpitch = run_gaze_estimation_model_tensorrt(img, face_bbox, model)

    else:
        raise NotImplementedError(f"{mode} mode not implemented for gaze estimation")

    if timer is not None:
        timer.stop("gaze estimation")

    return yawpitch


def get_gaze_endpoints(img, face_bb, gaze_yawpitch):
    gaze_head, point_along_gaze = get_point_along_gaze(face_bb, gaze_yawpitch)

    if gaze_head is None:
        return None, None

    h, w, _ = img.shape
    gaze_tail = find_edge_intersection(w, h, gaze_head, point_along_gaze)

    gaze_head, gaze_tail = (int(gaze_head[0]), int(gaze_head[1])), (
        int(gaze_tail[0]),
        int(gaze_tail[1]),
    )
    return gaze_head, gaze_tail
