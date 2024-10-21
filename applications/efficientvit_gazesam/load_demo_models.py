import os
import sys

import onnxruntime as ort
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.gazesamcore.utils.consts import ONNX, PYTORCH, TENSORRT
from efficientvit.gazesamcore.utils.load_engine import (
    load_depth_estimation_engine,
    load_evit_decoder_engine,
    load_evit_encoder_engine,
    load_face_detection_engine,
    load_gaze_estimation_engine,
    load_yolo_engine,
)

PYTORCH_MODEL_DIR = "models/pytorch"
TRT_MODEL_DIR = "models/tensorrt"
ONNX_MODEL_DIR = "models/onnx"
ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def load_face_detection_model(mode, precision):
    if mode == TENSORRT:
        model_path = f"{TRT_MODEL_DIR}/{precision}/face_detection_{precision}.engine"

        return load_face_detection_engine(model_path)

    elif mode == ONNX:
        model_path = f"{ONNX_MODEL_DIR}/face_detection.onnx"
        return ort.InferenceSession(model_path, providers=ONNX_PROVIDERS)

    elif mode == PYTORCH:
        from onnx2torch import convert  # PyTorch face detection model

        model_path = f"{ONNX_MODEL_DIR}/face_detection.onnx"
        model = convert(model_path)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")

        return model

    else:
        raise NotImplementedError(f"{mode} mode not implemented for face detection")


def load_gaze_estimation_model(mode, precision):
    if mode == TENSORRT:
        model_path = f"{TRT_MODEL_DIR}/{precision}/gaze_estimation_{precision}.engine"

        return load_gaze_estimation_engine(model_path)

    elif mode == ONNX:
        model_path = f"{ONNX_MODEL_DIR}/gaze_estimation.onnx"
        return ort.InferenceSession(model_path, providers=ONNX_PROVIDERS)

    elif mode == PYTORCH:
        from l2cs import Pipeline  # PyTorch gaze estimation model

        gaze_pipeline = Pipeline(
            weights=f"{PYTORCH_MODEL_DIR}/L2CSNet_gaze360.pkl",
            arch="ResNet50",
            device=torch.device("cuda"),
        )

        def get_gaze_yaw_pitch(frame):
            gaze_res = gaze_pipeline.step(frame)
            yaw = gaze_res.yaw[0]
            pitch = gaze_res.pitch[0]

            return pitch, yaw

        return get_gaze_yaw_pitch

    else:
        raise NotImplementedError(f"{mode} mode not implemented for gaze estimation")


def load_yolo_model(mode, precision):
    if mode == TENSORRT:
        model_path = f"{TRT_MODEL_DIR}/{precision}/yolo_m_{precision}.engine"

        return load_yolo_engine(model_path)

    elif mode == ONNX:
        model_path = f"{ONNX_MODEL_DIR}/yolo_m_ort.onnx"
        return ort.InferenceSession(model_path, providers=ONNX_PROVIDERS)

    elif mode == PYTORCH:
        from super_gradients.common.object_names import Models  # PyTorch YOLO model
        from super_gradients.training import models  # PyTorch YOLO model

        model = models.get(Models.YOLO_NAS_M, pretrained_weights="coco")
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        return model

    else:
        raise NotImplementedError(f"{mode} mode not implemented for YOLO")


def load_depth_model(mode, precision):
    if mode == TENSORRT:
        model_path = f"{TRT_MODEL_DIR}/{precision}/depth_m_{precision}.engine"

        return load_depth_estimation_engine(model_path)

    elif mode == ONNX:
        model_path = f"{ONNX_MODEL_DIR}/depth_m.onnx"
        return ort.InferenceSession(model_path, providers=ONNX_PROVIDERS)

    elif mode == PYTORCH:
        # make sure to follow instructions in README to download this model correctly
        import sys

        sys.path.append("Depth-Anything")
        from depth_anything.dpt import DepthAnything

        model_configs = {
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
        }

        encoder = "vitb"  # or 'vitb', 'vits'
        model = DepthAnything(model_configs[encoder])
        model.load_state_dict(torch.load(f"{PYTORCH_MODEL_DIR}/depth_anything_{encoder}14.pth", weights_only=True))
        model = model.eval().cuda()
        return model

    else:
        raise NotImplementedError(f"{mode} mode not implemented for depth estimation")


def load_evit_model(
    mode,
    model_type,
    encoder_precision,
    decoder_precision,
):
    if mode == TENSORRT:
        from efficientvit.gazesamcore.evit import TrtEvitSam

        evit_encoder_model_path = (
            f"{TRT_MODEL_DIR}/{encoder_precision}/evit_encoder_{model_type}_{encoder_precision}.engine"
        )
        evit_decoder_model_path = (
            f"{TRT_MODEL_DIR}/{decoder_precision}/evit_decoder_{model_type}_{decoder_precision}.engine"
        )

        evit_encoder = load_evit_encoder_engine(evit_encoder_model_path)
        evit_decoder = load_evit_decoder_engine(evit_decoder_model_path)

        model = TrtEvitSam(model_type, evit_encoder, evit_decoder)
        return model

    elif mode == ONNX:
        from efficientvit.gazesamcore.evit import OnnxEvitSam, OnnxEvitSamDecoder, OnnxEvitSamEncoder

        evit_encoder_model_path = f"{ONNX_MODEL_DIR}/evit_encoder_{model_type}.onnx"
        evit_decoder_model_path = f"{ONNX_MODEL_DIR}/evit_decoder_{model_type}.onnx"

        evit_encoder = OnnxEvitSamEncoder(evit_encoder_model_path)
        evit_decoder = OnnxEvitSamDecoder(evit_decoder_model_path)

        model = OnnxEvitSam(model_type, evit_encoder, evit_decoder)
        return model

    elif mode == PYTORCH:
        from efficientvit.gazesamcore.evit import PytorchEvitSam
        from efficientvit.sam_model_zoo import create_sam_model

        efficientvit_sam = create_sam_model(model_type, True).cuda().eval()

        return PytorchEvitSam(efficientvit_sam)

    else:
        raise NotImplementedError(f"{mode} mode not implemented for EfficientViT-SAM")
