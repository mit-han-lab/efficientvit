import argparse
import os
import sys
import time

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)


from load_demo_models import (
    load_depth_model,
    load_evit_model,
    load_face_detection_model,
    load_gaze_estimation_model,
    load_yolo_model,
)

from efficientvit.gazesamcore.depth import get_depth_map
from efficientvit.gazesamcore.evit import get_evit_masks
from efficientvit.gazesamcore.face import detect_face, get_face_bbox
from efficientvit.gazesamcore.gaze import estimate_gaze, get_gaze_endpoints
from efficientvit.gazesamcore.utils.draw import annotate_blank_frame, annotate_frame
from efficientvit.gazesamcore.utils.smoother import GazeSmoother, LandmarkSmoother, OneEuroFilter
from efficientvit.gazesamcore.utils.timer import Timer
from efficientvit.gazesamcore.yolo import filter_bboxes_depth, filter_bboxes_face, filter_bboxes_gaze, get_yolo_bboxes


def setup_video(webcam, input_vid, output_dir):
    if webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_vid)

    if not cap.isOpened():
        print(f"Error opening video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps:", fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    _, frame = cap.read()
    height, width, _ = frame.shape
    print("frame shape:", (width, height))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if webcam:
        output_vid = os.path.join(output_dir, "webcam_output.mp4")
    else:
        output_vid = os.path.join(output_dir, os.path.basename(input_vid))

    out = cv2.VideoWriter(output_vid, fourcc, fps, (width, height))

    return cap, out


def validate_args(args):
    if args.webcam and args.video:
        print("Error: Cannot use both webcam and video file input.")
        print("Either pass in --webcam or set --video <path>.")
        sys.exit(1)

    if not args.webcam and (not args.video or not os.path.exists(args.video)):
        print(f"Input video {args.video} does not exist")
        sys.exit(1)


def set_precisions(args):
    if args.evit_encoder_precision is None:
        args.evit_encoder_precision = "fp32"

    if args.evit_decoder_precision is None:
        args.evit_decoder_precision = "fp16"

    if args.face_detection_precision is None:
        args.face_detection_precision = "fp16"

    if args.depth_precision is None:
        args.depth_precision = "fp16"

    if args.gaze_estimation_precision is None:
        args.gaze_estimation_precision = "fp16" if args.precision_mode == "default" else "int8"

    if args.yolo_precision is None:
        args.yolo_precision = "fp16" if args.precision_mode == "default" else "int8"


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Get input from webcam",
    )

    parser.add_argument(
        "--video",
        help="Get input from video file.",
    )

    parser.add_argument(
        "--output-dir",
        default="output_videos",
        help="Output directory for GazeSAM annotated video.",
    )

    parser.add_argument(
        "--runtime",
        type=str,
        default="tensorrt",
        choices=["tensorrt", "onnx", "pytorch"],
        help="Recommended to only run TensorRT.  Other modes are purely for benchmarking.",
    )

    parser.add_argument(
        "--evit-model-type",
        type=str,
        default="efficientvit-sam-l0",
        choices=[
            "efficientvit-sam-l0",
            "efficientvit-sam-l1",
            "efficientvit-sam-l2",
            "efficientvit-sam-xl0",
            "efficientvit-sam-xl1",
        ],
        help="EfficientViT model type.  If use other than l0, follow README instructions for model compilation.",
    )

    parser.add_argument(
        "--precision-mode",
        default="default",
        choices=["default", "optimized"],
        help="Engine compilation mode used to generate models. 'default' uses the fp32+fp16 engines,  'optimized' uses the fp32+fp16+int8 engines.",
    )

    parser.add_argument(
        "--face-detection-precision",
        type=str,
        choices=["fp32", "fp16"],
        default=None,
        help="Precision of face detection model.",
    )

    parser.add_argument(
        "--gaze-estimation-precision",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default=None,
        help="Precision of gaze estimation model.",
    )

    parser.add_argument(
        "--yolo-precision",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default=None,
        help="Precision of YOLO model.",
    )

    parser.add_argument(
        "--depth-precision",
        type=str,
        choices=["fp32", "fp16"],
        default=None,
        help="Precision of depth model.",
    )

    parser.add_argument(
        "--evit-encoder-precision",
        type=str,
        choices=["fp32"],
        default=None,
        help="Precision of EfficientViT encoder model",
    )

    parser.add_argument(
        "--evit-decoder-precision",
        type=str,
        choices=["fp32", "fp16"],
        default=None,
        help="Precision of EfficientViT decoder model.",
    )

    args = parser.parse_args()

    validate_args(args)
    set_precisions(args)

    print(f"Running {args.webcam or args.video}")
    print(f"Runtime: {args.runtime}")
    if args.runtime == "tensorrt":
        print(
            f"Model precisions: {args.face_detection_precision}, {args.gaze_estimation_precision}, {args.yolo_precision}, {args.depth_precision}, {args.evit_encoder_precision}, {args.evit_decoder_precision}"
        )

    return args


def render_frame(out, frame, webcam, frame_times):
    fps = ""
    if len(frame_times) > 2:
        fps = f"RTX 4070: {int(1/frame_times[-1])} FPS"

    cv2.putText(
        frame,
        fps,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        thickness=2,
    )

    if webcam:
        cv2.imshow("demo", frame)

    out.write(frame)


def main(args):
    cap, out = setup_video(args.webcam, args.video, args.output_dir)

    bbox_smoother = LandmarkSmoother(OneEuroFilter, pt_num=2, min_cutoff=0.0, beta=1.0)
    gaze_smoother = GazeSmoother(OneEuroFilter, min_cutoff=0.01, beta=0.8)

    face_detection_model = load_face_detection_model(args.runtime, args.face_detection_precision)
    gaze_estimation_model = load_gaze_estimation_model(args.runtime, args.gaze_estimation_precision)
    yolo_model = load_yolo_model(args.runtime, args.yolo_precision)
    depth_model = load_depth_model(args.runtime, args.depth_precision)
    evit_model = load_evit_model(
        args.runtime,
        args.evit_model_type,
        args.evit_encoder_precision,
        args.evit_decoder_precision,
    )

    timer = Timer()
    frame_start_time = time.time()
    frame_times = []

    # store previous mask and bbox to use if no new mask is found
    saved_mask, saved_bbox = None, None

    try:
        while True:
            frame_times.append(time.time() - frame_start_time)
            frame_start_time = time.time()

            ret, frame = cap.read(0) if args.webcam else cap.read()
            if not ret or frame is None:
                break

            if cv2.waitKey(1) == 27:
                break

            faces = detect_face(frame, face_detection_model, args.runtime, timer)
            face_bbox = get_face_bbox(faces, bbox_smoother)

            if face_bbox is None:
                render_frame(out, frame, args.webcam, frame_times)
                saved_mask, saved_bbox = None, None
                continue

            gaze_yawpitch = estimate_gaze(frame, face_bbox, gaze_estimation_model, args.runtime, timer)

            if gaze_yawpitch is None:
                render_frame(out, frame, args.webcam, frame_times)
                saved_mask, saved_bbox = None, None
                continue

            gaze_yawpitch = gaze_smoother(gaze_yawpitch, t=time.time())
            gaze_head, gaze_tail = get_gaze_endpoints(frame, face_bbox, gaze_yawpitch)

            bboxes = get_yolo_bboxes(frame, yolo_model, args.runtime, timer)
            bboxes = filter_bboxes_face(bboxes, face_bbox)
            bboxes = filter_bboxes_gaze(bboxes, gaze_head, gaze_tail)

            if len(bboxes) == 0:
                frame = annotate_blank_frame(
                    frame,
                    gaze_head,
                    gaze_tail,
                    saved_mask,
                    saved_bbox,
                    args.runtime,
                    args.webcam,
                )
                render_frame(out, frame, args.webcam, frame_times)
                saved_mask, saved_bbox = None, None
                continue

            depth_map = get_depth_map(frame, depth_model, args.runtime, timer)

            bboxes, depth_mask = filter_bboxes_depth(bboxes, depth_map, gaze_head)
            depth_frame = (depth_mask.cpu().numpy() * 255).astype(np.uint8)
            depth_frame = np.stack((depth_frame,) * 3, axis=-1)

            if len(bboxes) == 0:
                frame = annotate_blank_frame(
                    frame,
                    gaze_head,
                    gaze_tail,
                    saved_mask,
                    saved_bbox,
                    args.runtime,
                    args.webcam,
                )
                render_frame(out, frame, args.webcam, frame_times)
                saved_mask, saved_bbox = None, None
                continue

            masks, iou_preds = get_evit_masks(frame, bboxes, evit_model, args.runtime, timer)
            frame, saved_mask, saved_bbox = annotate_frame(
                frame, gaze_head, gaze_tail, bboxes, masks, iou_preds, depth_mask
            )

            render_frame(out, frame, args.webcam, frame_times)

    finally:
        out.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = get_args()
    main(args)
