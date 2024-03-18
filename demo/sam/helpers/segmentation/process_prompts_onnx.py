from copy import deepcopy

from demo.sam.helpers.predictors.effvit_sam_onnx import OnnxEfficientViTSamPredictor
from demo.sam.helpers.auto_mask_generator import DemoAccelEfficientViTSamAutomaticMaskGenerator
from demo.sam.helpers.utils import (
    ONNX,
    get_point_inputs, 
    get_box_inputs, 
    draw_point_masks,
    draw_box_masks, 
    draw_point_and_box_masks, 
    draw_all_masks
)


def segment_using_points_onnx(prompt_dict, model_name):
    effvit_sam_predictor = OnnxEfficientViTSamPredictor(model_name) 
    
    raw_img, prompts = prompt_dict['image'], prompt_dict['points']
    origin_image_size = raw_img.shape[:2]

    points = get_point_inputs(prompts)
    if len(points) == 0:
        return raw_img
    
    point_coords = points[..., :2]
    original_point_coords = deepcopy(point_coords)

    effvit_sam_predictor.set_image(raw_img)

    masks, _ = effvit_sam_predictor.predict_torch(
        point_coords=point_coords,
        point_expansion_axis=0,
        im_size = origin_image_size)    

    masks = masks.cpu().numpy()  

    return draw_point_masks(raw_img, masks, original_point_coords)


def segment_using_boxes_onnx(prompt_dict, model_name):
    effvit_sam_predictor = OnnxEfficientViTSamPredictor(model_name) 
    
    raw_img, prompts = prompt_dict['image'], prompt_dict['points']
    origin_image_size = raw_img.shape[:2]

    boxes = get_box_inputs(prompts)
    original_boxes = deepcopy(boxes)
    if len(boxes) == 0:
        return raw_img
    
    effvit_sam_predictor.set_image(raw_img)
    
    masks, _ = effvit_sam_predictor.predict_torch(
        im_size=origin_image_size,
        boxes=boxes)
    
    masks = masks.cpu().numpy()

    return draw_box_masks(raw_img, masks, original_boxes)


def segment_using_points_and_boxes_onnx(prompt_dict, model_name):
    effvit_sam_predictor = OnnxEfficientViTSamPredictor(model_name) 
    
    raw_img, prompts = prompt_dict['image'], prompt_dict['points']
    origin_image_size = raw_img.shape[:2]

    boxes = get_box_inputs(prompts)
    points = get_point_inputs(prompts)

    if len(boxes) == 0 and len(points) == 0:
        return raw_img
    elif len(boxes) == 0:
        return segment_using_points_onnx(prompt_dict, model_name)
    elif len(points) == 0:
        return segment_using_boxes_onnx(prompt_dict, model_name)
    
    point_coords = points[..., :2]
    original_boxes = deepcopy(boxes)
    original_point_coords = deepcopy(point_coords)

    effvit_sam_predictor.set_image(raw_img)

    masks, _ = effvit_sam_predictor.predict_torch(
        im_size=origin_image_size,
        point_coords=point_coords,
        point_expansion_axis=0,
        boxes=boxes)
    
    masks = masks.cpu().numpy()

    return draw_point_and_box_masks(raw_img, masks, original_point_coords, original_boxes)


def segment_full_img_onnx(raw_image, model_name, points_per_batch, pred_iou_thresh, stability_score_thresh, box_nms_thresh):
    effvit_mask_gen = DemoAccelEfficientViTSamAutomaticMaskGenerator(mode=ONNX, model_name=model_name)

    effvit_mask_gen.set_points_per_batch(points_per_batch)
    effvit_mask_gen.set_pred_iou_thresh(pred_iou_thresh)
    effvit_mask_gen.set_stability_score_thresh(stability_score_thresh)
    effvit_mask_gen.set_box_nms_thresh(box_nms_thresh)

    anns = effvit_mask_gen.generate(raw_image)
    return draw_all_masks(raw_image, anns)
