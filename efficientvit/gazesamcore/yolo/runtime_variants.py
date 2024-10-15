import cv2
import numpy as np
import onnxruntime as ort
import torch

__all__ = ["run_yolo_model_onnx", "run_yolo_model_tensorrt"]


def run_yolo_model_onnx(img, session):
    img = cv2.resize(img, (640, 640))
    img_bchw = np.transpose(np.expand_dims(img, 0), (0, 3, 1, 2))
    inputs = [o.name for o in session.get_inputs()]
    outputs = [o.name for o in session.get_outputs()]

    io_binding = session.io_binding()
    input_ort = ort.OrtValue.ortvalue_from_numpy(img_bchw, "cuda", 0)
    io_binding.bind_input(
        name=inputs[0],
        device_type=input_ort.device_name(),
        device_id=0,
        element_type=np.uint8,
        shape=input_ort.shape(),
        buffer_ptr=input_ort.data_ptr(),
    )

    io_binding.bind_output(outputs[0])
    io_binding.bind_output(outputs[1])
    io_binding.bind_output(outputs[2])
    io_binding.bind_output(outputs[3])

    session.run_with_iobinding(io_binding)
    pred = io_binding.copy_outputs_to_cpu()
    pred_bboxes = pred[1][0]

    return pred_bboxes


def run_yolo_model_tensorrt(img, model):
    img = cv2.resize(img, (640, 640))
    img = np.transpose(np.expand_dims(img, axis=0), (0, 3, 1, 2))

    img = torch.tensor(img).cuda()
    preds = model(img)
    pred_bboxes = preds[1][0]
    pred_bboxes = pred_bboxes.cpu().numpy()

    return pred_bboxes
