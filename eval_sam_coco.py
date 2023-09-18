# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import argparse
import inspect
import os
import sys

import numpy as np
from torchvision.datasets import CocoDetection
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_sam_model


def bbox_xywh_to_xyxy(bbox: list[int]) -> list[int]:
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    return float(intersection / union) * 100


def predict_mask(predictor: EfficientViTSamPredictor, bbox: list[int]) -> np.ndarray:
    masks, iou_predictions, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(bbox),
        multimask_output=True,
    )

    mask = masks[iou_predictions.argmax()]
    return mask


def filter_results_by_area(results: list[dict], min=None, max=None) -> list[dict]:
    filtered = []
    for r in results:
        if min is not None and r["area"] < min:
            continue
        if max is not None and r["area"] > max:
            continue
        filtered.append(r)
    return filtered


def get_coco_metric(results: list[dict]) -> dict[str, float]:
    small_results = filter_results_by_area(results, None, 32**2)
    medium_results = filter_results_by_area(results, 32**2, 96**2)
    large_results = filter_results_by_area(results, 96**2, None)

    return {
        "all": sum(r["iou"] for r in results) / len(results),
        "large": sum(r["iou"] for r in large_results) / len(large_results),
        "medium": sum(r["iou"] for r in medium_results) / len(medium_results),
        "small": sum(r["iou"] for r in small_results) / len(small_results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="/dataset/coco/val2017")
    parser.add_argument("--anno_path", type=str, default="/dataset/coco/annotations/instances_val2017.json")
    parser.add_argument("--model", type=str)
    parser.add_argument("--weight_url", type=str, default=None)

    args = parser.parse_args()

    # dataset
    dataset = CocoDetection(
        root=args.image_path,
        annFile=args.anno_path,
    )

    # build model
    efficientvit_sam = create_sam_model(args.model, True, args.weight_url).cuda().eval()
    efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)

    # run
    results = []
    with tqdm(total=len(dataset)) as t:
        for i in range(len(dataset)):
            image, anns = dataset[i]
            image = np.array(image)
            efficientvit_sam_predictor.set_image(image)
            for ann in anns:
                bbox = bbox_xywh_to_xyxy(ann["bbox"])
                mask = dataset.coco.annToMask(ann)
                mask_coco = mask > 0
                mask_sam = predict_mask(efficientvit_sam_predictor, bbox)

                miou = iou(mask_sam, mask_coco)
                result = {
                    "id": ann["id"],
                    "area": ann["area"],
                    "category_id": ann["category_id"],
                    "iscrowd": ann["iscrowd"],
                    "image_id": ann["image_id"],
                    "box": bbox,
                    "iou": miou,
                }

                results.append(result)

            t.set_postfix(get_coco_metric(results))
            t.update()
    print(", ".join([f"{key}={val:.1f}" for key, val in get_coco_metric(results).items()]))


if __name__ == "__main__":
    main()
