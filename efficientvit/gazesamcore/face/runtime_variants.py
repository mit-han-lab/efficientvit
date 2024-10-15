import numpy as np
import onnxruntime as ort
import torch

__all__ = ["run_face_detection_model_pytorch", "run_face_detection_model_onnx", "run_face_detection_model_trt"]


@torch.inference_mode()
def run_face_detection_model_pytorch(img, model):
    img = torch.tensor(img).cuda()
    output = model(img).cpu().numpy()
    return output


def run_face_detection_model_onnx(img, session):
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

    return output[0]


def run_face_detection_model_trt(img, model):
    img = torch.tensor(img).cuda()
    output = model(img)
    output = output.cpu().numpy()
    return output
