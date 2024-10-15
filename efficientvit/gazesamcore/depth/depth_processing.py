import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from efficientvit.gazesamcore.utils import ONNX, PYTORCH, TENSORRT

__all__ = ["get_depth_map", "depth_preprocess", "run_depth_model_onnx"]


def get_depth_map(img, model, mode, timer=None):
    orig_shape = img.shape[:2]
    img = depth_preprocess(img)

    if timer is not None:
        timer.start("depth estimation")

    if mode == PYTORCH:
        img = torch.tensor(img).cuda()
        with torch.no_grad():
            depth = model(img)
        depth = depth[None]

    elif mode == ONNX:
        depth = run_depth_model_onnx(img, model)
        depth = torch.tensor(depth).cuda()

    elif mode == TENSORRT:
        img = torch.tensor(img).cuda()
        depth = model(img)

    else:
        raise NotImplementedError(f"{mode} mode not implemented for depth estimation")

    if timer is not None:
        timer.stop("depth estimation")

    depth = F.interpolate(depth, size=orig_shape)[0][0]
    depth_map = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    return depth_map


def depth_preprocess(img):
    preprocessed_img = cv2.resize(img, (518, 518), interpolation=cv2.INTER_CUBIC)
    preprocessed_img = preprocessed_img / 255.0
    preprocessed_img = np.transpose(preprocessed_img, (2, 0, 1))
    preprocessed_img = np.ascontiguousarray(preprocessed_img).astype(np.float32)
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
    return preprocessed_img


def run_depth_model_onnx(img, session):
    outputs = [o.name for o in session.get_outputs()]

    io_binding = session.io_binding()
    input_ort = ort.OrtValue.ortvalue_from_numpy(img, "cuda", 0)
    io_binding.bind_input(
        name=session.get_inputs()[0].name,
        device_type=input_ort.device_name(),
        device_id=0,
        element_type=np.float32,
        shape=input_ort.shape(),
        buffer_ptr=input_ort.data_ptr(),
    )

    io_binding.bind_output(outputs[0])
    session.run_with_iobinding(io_binding)

    output = io_binding.copy_outputs_to_cpu()
    depth = output[0]
    return depth
