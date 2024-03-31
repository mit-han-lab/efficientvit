import numpy as np
import torch

from demo.sam.helpers.auto_mask_generator import DemoEfficientViTSamAutomaticMaskGenerator
from demo.sam.helpers.predictors.effvit_sam_pytorch import PyTorchEfficientViTSamPredictor
from demo.sam.helpers.utils import (
    MUTLIMASK,
    draw_all_masks,
    draw_box_masks,
    draw_point_and_box_masks,
    draw_point_masks,
    get_box_inputs,
    get_point_inputs,
)
from efficientvit.sam_model_zoo import create_sam_model

get_weight_url = lambda model_name: f"assets/checkpoints/sam/{model_name}.pt"


def get_predictor(model_name):
    weight_url = get_weight_url(model_name)

    if torch.cuda.is_available():
        efficientvit_sam = create_sam_model(model_name, True, weight_url).cuda().eval()
    else:
        efficientvit_sam = create_sam_model(model_name, True, weight_url).eval()

    return PyTorchEfficientViTSamPredictor(efficientvit_sam)


def get_full_mask_generator(model_name):
    weight_url = get_weight_url(model_name)

    if torch.cuda.is_available():
        efficientvit_sam = create_sam_model(model_name, True, weight_url).cuda().eval()
    else:
        efficientvit_sam = create_sam_model(model_name, True, weight_url).eval()

    return DemoEfficientViTSamAutomaticMaskGenerator(efficientvit_sam)


def segment_using_points_pytorch(prompt_dict, model_name):
    efficientvit_sam_predictor = get_predictor(model_name)
    raw_image, prompts = prompt_dict["image"], prompt_dict["points"]
    points = get_point_inputs(prompts)

    if len(points) == 0:
        return raw_image

    point_coords = points[..., :2]
    point_labels = points[..., 2]

    efficientvit_sam_predictor.set_image(raw_image)
    masks, _, _ = efficientvit_sam_predictor.predict(
        point_coords=np.array(point_coords),
        point_labels=np.array(point_labels),
        multimask_output=MUTLIMASK,
    )

    return draw_point_masks(raw_image, masks, points)


def segment_using_boxes_pytorch(prompt_dict, model_name):
    efficientvit_sam_predictor = get_predictor(model_name)
    raw_image, prompts = prompt_dict["image"], prompt_dict["points"]
    boxes = np.array(get_box_inputs(prompts))

    if len(boxes) == 0:
        return raw_image

    efficientvit_sam_predictor.set_image(raw_image)
    masks, _, _ = efficientvit_sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        boxes=boxes,
        multimask_output=MUTLIMASK,
    )

    return draw_box_masks(raw_image, masks, boxes)


def segment_using_points_and_boxes_pytorch(prompt_dict, model_name):
    efficientvit_sam_predictor = get_predictor(model_name)
    raw_image, prompts = prompt_dict["image"], prompt_dict["points"]

    boxes = get_box_inputs(prompts)
    points = get_point_inputs(prompts)

    if len(boxes) == 0 and len(points) == 0:
        return raw_image
    elif len(boxes) == 0:
        return segment_using_points_pytorch(prompt_dict, model_name)
    elif len(points) == 0:
        return segment_using_boxes_pytorch(prompt_dict, model_name)

    point_coords = points[..., :2]
    point_labels = points[..., 2]
    boxes = np.array(boxes)

    efficientvit_sam_predictor.set_image(raw_image)
    masks, _, _ = efficientvit_sam_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        boxes=boxes,
        multimask_output=MUTLIMASK,
    )

    return draw_point_and_box_masks(raw_image, masks, points, boxes)


def segment_full_img_pytorch(
    raw_image, model_name, points_per_batch, pred_iou_thresh, stability_score_thresh, box_nms_thresh
):
    effvit_mask_gen = get_full_mask_generator(model_name)
    effvit_mask_gen.set_points_per_batch(points_per_batch)
    effvit_mask_gen.set_pred_iou_thresh(pred_iou_thresh)
    effvit_mask_gen.set_stability_score_thresh(stability_score_thresh)
    effvit_mask_gen.set_box_nms_thresh(box_nms_thresh)

    anns = effvit_mask_gen.generate(raw_image)
    return draw_all_masks(raw_image, anns)
