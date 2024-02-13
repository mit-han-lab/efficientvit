# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from PIL import Image

from efficientvit.apps.utils import parse_unknown_args
from efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator, EfficientViTSamPredictor
from efficientvit.models.utils import build_kwargs_from_config
from efficientvit.sam_model_zoo import create_sam_model


def load_image(data_path: str, mode="rgb") -> np.ndarray:
    img = Image.open(data_path)
    if mode == "rgb":
        img = img.convert("RGB")
    return np.array(img)


def cat_images(image_list: list[np.ndarray], axis=1, pad=20) -> np.ndarray:
    shape_list = [image.shape for image in image_list]
    max_h = max([shape[0] for shape in shape_list]) + pad * 2
    max_w = max([shape[1] for shape in shape_list]) + pad * 2

    for i, image in enumerate(image_list):
        canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        h, w, _ = image.shape
        crop_y = (max_h - h) // 2
        crop_x = (max_w - w) // 2
        canvas[crop_y : crop_y + h, crop_x : crop_x + w] = image
        image_list[i] = canvas

    image = np.concatenate(image_list, axis=axis)
    return image


def show_anns(anns) -> None:
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]["segmentation"].shape[0], sorted_anns[0]["segmentation"].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def draw_binary_mask(raw_image: np.ndarray, binary_mask: np.ndarray, mask_color=(0, 0, 255)) -> np.ndarray:
    color_mask = np.zeros_like(raw_image, dtype=np.uint8)
    color_mask[binary_mask == 1] = mask_color
    mix = color_mask * 0.5 + raw_image * (1 - 0.5)
    binary_mask = np.expand_dims(binary_mask, axis=2)
    canvas = binary_mask * mix + (1 - binary_mask) * raw_image
    canvas = np.asarray(canvas, dtype=np.uint8)
    return canvas


def draw_bbox(
    image: np.ndarray,
    bbox: list[list[int]],
    color: str or list[str] = "g",
    linewidth=1,
    tmp_name=".tmp.png",
) -> np.ndarray:
    dpi = 300
    oh, ow, _ = image.shape
    plt.close()
    plt.figure(1, figsize=(oh / dpi, ow / dpi))
    plt.imshow(image)
    if isinstance(color, str):
        color = [color for _ in bbox]
    for (x0, y0, x1, y1), c in zip(bbox, color):
        plt.gca().add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, lw=linewidth, edgecolor=c, facecolor=(0, 0, 0, 0)))
    plt.axis("off")
    plt.savefig(tmp_name, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    image = cv2.resize(load_image(tmp_name), dsize=(ow, oh))
    os.remove(tmp_name)
    plt.close()
    return image


def draw_scatter(
    image: np.ndarray,
    points: list[list[int]],
    color: str or list[str] = "g",
    marker="*",
    s=10,
    ew=0.25,
    tmp_name=".tmp.png",
) -> np.ndarray:
    dpi = 300
    oh, ow, _ = image.shape
    plt.close()
    plt.figure(1, figsize=(oh / dpi, ow / dpi))
    plt.imshow(image)
    if isinstance(color, str):
        color = [color for _ in points]
    for (x, y), c in zip(points, color):
        plt.scatter(x, y, color=c, marker=marker, s=s, edgecolors="white", linewidths=ew)
    plt.axis("off")
    plt.savefig(tmp_name, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    image = cv2.resize(load_image(tmp_name), dsize=(ow, oh))
    os.remove(tmp_name)
    plt.close()
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--multimask", action="store_true")
    parser.add_argument("--image_path", type=str, default="assets/fig/cat.jpg")
    parser.add_argument("--output_path", type=str, default="assets/demo/efficientvit_sam_demo.png")

    parser.add_argument("--mode", type=str, default="all", choices=["point", "box", "all"])
    parser.add_argument("--point", type=str, default=None)
    parser.add_argument("--box", type=str, default=None)

    # EfficientViTSamAutomaticMaskGenerator args
    parser.add_argument("--pred_iou_thresh", type=float, default=0.8)
    parser.add_argument("--stability_score_thresh", type=float, default=0.85)
    parser.add_argument("--min_mask_region_area", type=float, default=100)

    args, opt = parser.parse_known_args()
    opt = parse_unknown_args(opt)

    # build model
    efficientvit_sam = create_sam_model(args.model, True, args.weight_url).cuda().eval()
    efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)
    efficientvit_mask_generator = EfficientViTSamAutomaticMaskGenerator(
        efficientvit_sam,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
        **build_kwargs_from_config(opt, EfficientViTSamAutomaticMaskGenerator),
    )

    # load image
    raw_image = np.array(Image.open(args.image_path).convert("RGB"))
    H, W, _ = raw_image.shape
    print(f"Image Size: W={W}, H={H}")

    tmp_file = f".tmp_{time.time()}.png"
    if args.mode == "all":
        masks = efficientvit_mask_generator.generate(raw_image)
        plt.figure(figsize=(20, 20))
        plt.imshow(raw_image)
        show_anns(masks)
        plt.axis("off")
        plt.savefig(args.output_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.0)
    elif args.mode == "point":
        args.point = yaml.safe_load(args.point or f"[[{W // 2},{H // 2},{1}]]")
        point_coords = [(x, y) for x, y, _ in args.point]
        point_labels = [l for _, _, l in args.point]

        efficientvit_sam_predictor.set_image(raw_image)
        masks, _, _ = efficientvit_sam_predictor.predict(
            point_coords=np.array(point_coords),
            point_labels=np.array(point_labels),
            multimask_output=args.multimask,
        )
        plots = [
            draw_scatter(
                draw_binary_mask(raw_image, binary_mask, (0, 0, 255)),
                point_coords,
                color=["g" if l == 1 else "r" for l in point_labels],
                s=10,
                ew=0.25,
                tmp_name=tmp_file,
            )
            for binary_mask in masks
        ]
        plots = cat_images(plots, axis=1)
        Image.fromarray(plots).save(args.output_path)
    elif args.mode == "box":
        args.box = yaml.safe_load(args.box)
        efficientvit_sam_predictor.set_image(raw_image)
        masks, _, _ = efficientvit_sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(args.box),
            multimask_output=args.multimask,
        )
        plots = [
            draw_bbox(
                draw_binary_mask(raw_image, binary_mask, (0, 0, 255)),
                [args.box],
                color="g",
                tmp_name=tmp_file,
            )
            for binary_mask in masks
        ]
        plots = cat_images(plots, axis=1)
        Image.fromarray(plots).save(args.output_path)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
