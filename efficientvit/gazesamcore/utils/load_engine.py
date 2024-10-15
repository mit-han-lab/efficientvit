try:
    import tensorrt as trt
    from torch2trt import TRTModule
except Exception as e:
    print(f"Skipping tensorrt-runtime import error: {e}")
    print("If using a non-tensorrt runtime, ignore.  Otherwise, please ensure tensorrt and torch2trt are installed")
    pass

__all__ = [
    "load_engine",
    "load_evit_encoder_engine",
    "load_evit_decoder_engine",
    "load_face_detection_engine",
    "load_depth_estimation_engine",
    "load_gaze_estimation_engine",
    "load_yolo_engine",
]


def load_engine(path: str, input_names: list[str], output_names: list[str]):
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        trt.init_libnvinfer_plugins(logger, namespace="")
        with open(path, "rb") as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    image_encoder_trt = TRTModule(engine=engine, input_names=input_names, output_names=output_names)

    return image_encoder_trt


def load_evit_encoder_engine(path: str):
    input_names = ["input_image"]
    output_names = ["image_embeddings"]

    return load_engine(path, input_names, output_names)


def load_evit_decoder_engine(path: str):
    input_names = ["image_embeddings", "boxes"]
    output_names = ["low_res_masks", "iou_predictions"]

    return load_engine(path, input_names, output_names)


def load_face_detection_engine(path: str):
    input_names = ["full_image"]
    output_names = ["bbox_det"]

    return load_engine(path, input_names, output_names)


def load_depth_estimation_engine(path: str):
    input_names = ["input"]
    output_names = ["output"]

    return load_engine(path, input_names, output_names)


def load_gaze_estimation_engine(path: str):
    input_names = ["input"]
    output_names = ["gaze_yaw_pitch"]

    return load_engine(path, input_names, output_names)


def load_yolo_engine(path: str):
    input_names = ["input"]
    output_names = ["num_dets", "det_boxes", "det_scores", "det_classes"]

    return load_engine(path, input_names, output_names)
