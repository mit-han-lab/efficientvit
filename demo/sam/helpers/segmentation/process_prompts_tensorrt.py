from copy import deepcopy

import numpy as np

from demo.sam.helpers.utils import (
    TENSORRT,
    draw_all_masks,
    draw_box_masks,
    draw_point_and_box_masks,
    draw_point_masks,
    get_box_inputs,
    get_point_inputs,
)

try:
    from demo.sam.helpers.auto_mask_generator import DemoAccelEfficientViTSamAutomaticMaskGenerator
    from demo.sam.helpers.predictors.effvit_sam_tensorrt import TRTEfficientViTSamPredictor
    from deployment.sam.tensorrt.inference import SamResize
except Exception as e:
    print(f"Skipping tensorrt-runtime import error: {e}")
    print("If using a non-tensorrt runtime, ignore.  Otherwise, please ensure tensorrt and torch2trt are installed")
    pass


DIR = "assets/export_models/sam/tensorrt"
get_encoder_path = lambda model_name: f"{DIR}/{model_name}_encoder.engine"
get_point_decoder_path = lambda model_name: f"{DIR}/{model_name}_point_decoder.engine"
get_box_decoder_path = lambda model_name: f"{DIR}/{model_name}_box_decoder.engine"
get_point_and_box_decoder_path = lambda model_name: f"{DIR}/{model_name}_point_decoder.engine"
get_full_img_decoder_path = lambda model_name: f"{DIR}/{model_name}_full_img_decoder.engine"


def segment_using_points_tensorrt(prompt_dict, model_name):
    encoder_engine_path = get_encoder_path(model_name)
    decoder_engine_path = get_point_decoder_path(model_name)

    raw_image, prompts = prompt_dict["image"], prompt_dict["points"]
    points = get_point_inputs(prompts)

    if len(points) == 0:
        return raw_image

    original_points = deepcopy(points)
    point_coords = points[..., :2]
    point_labels = points[..., 2]

    origin_image_size = raw_image.shape[:2]
    input_size = SamResize.get_preprocess_shape(*origin_image_size, long_side_length=1024)

    effvit_sam_predictor = TRTEfficientViTSamPredictor(model_name, encoder_engine_path, decoder_engine_path)
    effvit_sam_predictor.set_image(raw_image)
    masks, _ = effvit_sam_predictor.predict_torch(
        im_size=input_size, point_coords=point_coords, point_labels=point_labels, point_expansion_axis=0
    )

    masks = masks.cpu().numpy()

    return draw_point_masks(raw_image, masks, original_points)


def segment_using_boxes_tensorrt(prompt_dict, model_name):
    encoder_engine_path = get_encoder_path(model_name)
    decoder_engine_path = get_box_decoder_path(model_name)

    raw_image, prompts = prompt_dict["image"], prompt_dict["points"]
    origin_image_size = raw_image.shape[:2]
    input_size = SamResize.get_preprocess_shape(*origin_image_size, long_side_length=1024)

    boxes = get_box_inputs(prompts)
    original_boxes = deepcopy(boxes)
    if len(boxes) == 0:
        return raw_image

    effvit_sam_predictor = TRTEfficientViTSamPredictor(model_name, encoder_engine_path, decoder_engine_path)
    effvit_sam_predictor.set_image(raw_image)
    masks, _ = effvit_sam_predictor.predict_torch(im_size=input_size, boxes=boxes)

    masks = masks.cpu().numpy()

    return draw_box_masks(raw_image, masks, original_boxes)


def segment_using_points_and_boxes_tensorrt(prompt_dict, model_name):
    encoder_engine_path = get_encoder_path(model_name)
    decoder_engine_path = get_point_and_box_decoder_path(model_name)

    raw_image, prompts = prompt_dict["image"], prompt_dict["points"]
    origin_image_size = raw_image.shape[:2]
    input_size = SamResize.get_preprocess_shape(*origin_image_size, long_side_length=1024)

    boxes = get_box_inputs(prompts)
    original_boxes = deepcopy(boxes)
    points = get_point_inputs(prompts)

    if len(boxes) == 0 and len(points) == 0:
        return raw_image
    elif len(boxes) == 0:
        return segment_using_points_tensorrt(prompt_dict, model_name)
    elif len(points) == 0:
        return segment_using_boxes_tensorrt(prompt_dict, model_name)

    input_size = SamResize.get_preprocess_shape(*origin_image_size, long_side_length=1024)

    point_coords = points[..., :2]
    point_labels = points[..., 2]
    original_points = deepcopy(points)

    effvit_sam_predictor = TRTEfficientViTSamPredictor(model_name, encoder_engine_path, decoder_engine_path)
    effvit_sam_predictor.set_image(raw_image)
    masks, _ = effvit_sam_predictor.predict_torch(
        im_size=input_size, point_coords=point_coords, point_labels=point_labels, point_expansion_axis=0, boxes=boxes
    )

    masks = masks.cpu().numpy()

    return draw_point_and_box_masks(raw_image, masks, original_points, original_boxes)


def segment_full_img_tensorrt(
    raw_image, model_name, points_per_batch, pred_iou_thresh, stability_score_thresh, box_nms_thresh
):
    encoder_engine_path = get_encoder_path(model_name)
    decoder_engine_path = get_full_img_decoder_path(model_name)

    effvit_mask_gen = DemoAccelEfficientViTSamAutomaticMaskGenerator(
        mode=TENSORRT,
        model_name=model_name,
        encoder_engine_path=encoder_engine_path,
        decoder_engine_path=decoder_engine_path,
    )

    effvit_mask_gen.set_points_per_batch(points_per_batch)
    effvit_mask_gen.set_pred_iou_thresh(pred_iou_thresh)
    effvit_mask_gen.set_stability_score_thresh(stability_score_thresh)
    effvit_mask_gen.set_box_nms_thresh(box_nms_thresh)

    anns = effvit_mask_gen.generate(raw_image)
    return draw_all_masks(raw_image, anns)
