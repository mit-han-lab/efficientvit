import numpy as np
import onnxruntime as ort
import torch

from efficientvit.gazesamcore.gaze.helpers import preprocess_gaze

__all__ = ["run_gaze_estimation_model_onnx", "run_gaze_estimation_model_tensorrt", "get_cropped_face"]


def run_gaze_estimation_model_onnx(img, face_bbox, model, timer=None):
    cropped_face = get_cropped_face(img, face_bbox)
    if cropped_face is None:
        return None

    preprocessed_img = preprocess_gaze([cropped_face])
    outputs = [o.name for o in model.get_outputs()]

    if timer is not None:
        timer.start("gaze estimation model")

    io_binding = model.io_binding()
    input_ort = ort.OrtValue.ortvalue_from_numpy(preprocessed_img, "cuda", 0)
    io_binding.bind_input(
        name=model.get_inputs()[0].name,
        device_type=input_ort.device_name(),
        device_id=0,
        element_type=np.float32,
        shape=input_ort.shape(),
        buffer_ptr=input_ort.data_ptr(),
    )

    io_binding.bind_output(outputs[0])
    model.run_with_iobinding(io_binding)
    output = io_binding.copy_outputs_to_cpu()

    if timer is not None:
        timer.stop("gaze estimation model")

    return output[0][0]


def run_gaze_estimation_model_tensorrt(img, face_bbox, model):
    cropped_face = get_cropped_face(img, face_bbox)
    if cropped_face is None:
        return None

    preprocessed_img = preprocess_gaze([cropped_face])
    preprocessed_img = torch.tensor(preprocessed_img, dtype=torch.float32, device="cuda")

    output = model(preprocessed_img)
    output = output.cpu().numpy()
    return output[0]


def get_cropped_face(img, face_bbox):
    frame_shape = img.shape[:2]
    x1, y1, x2, y2 = face_bbox
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame_shape[0]), min(y2, frame_shape[1])
    cropped_face = img[y1:y2, x1:x2]

    if x1 >= x2 or y1 >= y2:
        return None

    return cropped_face
