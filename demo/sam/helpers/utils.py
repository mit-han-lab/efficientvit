import os
import random
import re

import cv2
import numpy as np
import torch
from PIL import Image

YELLOW = (255, 244, 79)
GREY = (128, 128, 128)
RED = (255, 0, 0)
BLUE = (135, 206, 235)
PINK = (239, 149, 186)
BLACK = (0, 0, 0)

MUTLIMASK = True
MIN_MASK_REGION_AREA = 100

POINTS_PER_BATCH = 64
PRED_IOU_THRESH = 0.8
STABILITY_SCORE_THRESH = 0.85
BOX_NMS_THRESH = 0.70

PYTORCH = "pytorch"
ONNX = "onnx"
TENSORRT = "tensorrt"


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_available_models(runtime):
    model_priority = {"xl1": 1, "xl0": 2, "l2": 3, "l1": 4, "l0": 5}

    model_storage_loc = {
        PYTORCH: "assets/checkpoints/sam",
        ONNX: "assets/export_models/sam/onnx",
        TENSORRT: "assets/export_models/sam/tensorrt",
    }

    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
    model_dir = os.path.join(root_dir, model_storage_loc[runtime])

    # get runtime-specific model files
    model_files = []
    if os.path.exists(model_dir):
        model_files = os.listdir(model_dir)

    # extract all model types found, keep only known model types
    model_types = [re.split(r"[_.]", file)[0] for file in model_files]
    model_types = [model for model in model_types if model in model_priority.keys()]

    if runtime == PYTORCH:
        required_suffixes = [".pt"]

    elif runtime == ONNX:
        required_suffixes = ["_encoder.onnx", "_decoder.onnx"]

    elif runtime == TENSORRT:
        required_suffixes = [
            "_encoder.engine",
            "_point_decoder.engine",
            "_box_decoder.engine",
            "_full_img_decoder.engine",
        ]

    # ensure each model type has requisite set of runtime-specific model files needed for inference
    valid_model_type = lambda model: all([f"{model}{suffix}" in model_files for suffix in required_suffixes])
    available_models = set([model for model in model_types if valid_model_type(model)])

    # models listed in decreasing accuracy order
    return sorted(available_models, key=lambda model: model_priority[model])


def get_point_inputs(prompts):
    point_inputs = []
    for prompt in prompts:
        if prompt[5] == 4.0:
            point_inputs.append(prompt[:3])

    return np.array(point_inputs)


def get_box_inputs(prompts):
    box_inputs = []
    for prompt in prompts:
        if prompt[2] == 2.0 and prompt[5] == 3.0:
            box_inputs.append((prompt[0], prompt[1], prompt[3], prompt[4]))

    return np.array(box_inputs)


def draw_point_masks(img, masks, coord_and_label):
    fine_grained_mask = masks[0][-1]

    oh, ow = fine_grained_mask.shape
    img = draw_binary_mask(img, fine_grained_mask, mask_color=YELLOW)

    point_radius = ow // 125
    border_thickness = point_radius // 3

    for x, y, l in coord_and_label:
        point_color = BLUE if l == 1 else PINK
        cv2.circle(img, (int(x), int(y)), point_radius + border_thickness, BLACK, -1)
        cv2.circle(img, (int(x), int(y)), point_radius, point_color, -1)

    return img


def draw_box_masks(img, masks, boxes):
    for mask in masks:
        fine_grained_mask = mask[-1]
        mask_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = draw_binary_mask(img, fine_grained_mask, mask_color)

    for box in boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), RED, 2)

    return img


def draw_point_and_box_masks(img, masks, coord_and_label, boxes):
    if boxes is None:
        return draw_point_masks(img, masks, coord_and_label)
    if coord_and_label is None:
        return draw_box_masks(img, masks, boxes)

    masks = masks[0]
    _, _, ow = masks.shape

    fine_grained_mask = masks[-1]
    mask_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img = draw_binary_mask(img, fine_grained_mask, mask_color)

    for box in boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), RED, 2)

    point_radius = ow // 125
    border_thickness = point_radius // 3

    for x, y, l in coord_and_label:
        point_color = BLUE if l == 1 else PINK
        cv2.circle(img, (int(x), int(y)), point_radius + border_thickness, BLACK, -1)
        cv2.circle(img, (int(x), int(y)), point_radius, point_color, -1)

    return img


def draw_binary_mask(raw_image, binary_mask, mask_color):
    color_mask = np.zeros_like(raw_image, dtype=np.uint8)
    color_mask[binary_mask == 1] = mask_color
    mix = color_mask * 0.5 + raw_image * (1 - 0.5)
    binary_mask = np.expand_dims(binary_mask, axis=2)
    canvas = binary_mask * mix + (1 - binary_mask) * raw_image
    canvas = np.asarray(canvas, dtype=np.uint8)
    return canvas


def draw_all_masks(raw_image, anns) -> np.ndarray:
    if len(anns) == 0:
        return raw_image

    masked_image = raw_image
    for ann in anns:
        m = ann["segmentation"]
        mask_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        masked_image = draw_binary_mask(masked_image, m, mask_color)

    return masked_image
