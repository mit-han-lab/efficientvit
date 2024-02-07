# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import copy

import cv2
import numpy as np
from lvis import LVISEval, LVISResults
from pycocotools.cocoeval import COCOeval


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    return float(intersection / (union + 1e-7)) * 100


def filter_results_by_area(results: list[dict], min=None, max=None) -> list[dict]:
    filtered = []
    for r in results:
        if min is not None and r["area"] < min:
            continue
        if max is not None and r["area"] > max:
            continue
        filtered.append(r)
    return filtered


def get_iou_metric(results: list[dict]) -> dict[str, float]:
    small_results = filter_results_by_area(results, None, 32**2)
    medium_results = filter_results_by_area(results, 32**2, 96**2)
    large_results = filter_results_by_area(results, 96**2, None)

    return {
        "all": sum(r["iou"] for r in results) / len(results),
        "large": sum(r["iou"] for r in large_results) / len(large_results),
        "medium": sum(r["iou"] for r in medium_results) / len(medium_results),
        "small": sum(r["iou"] for r in small_results) / len(small_results),
    }


def evaluate_predictions_on_coco(
    coco_gt,
    coco_results,
    iou_type,
    cocoeval_fn=COCOeval,
    img_ids=None,
    max_dets_per_image=None,
):
    """
    Modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/coco_evaluation.py.
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)

        for c in coco_results:
            c.pop("bbox", None)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = cocoeval_fn(coco_gt, coco_dt, iou_type)

    if max_dets_per_image is None:
        max_dets_per_image = [1, 10, 100]  # Default from COCOEval
    else:
        assert (
            len(max_dets_per_image) >= 3
        ), "COCOeval requires maxDets (and max_dets_per_image) to have length at least 3"

        if max_dets_per_image[2] != 100:
            raise NotImplementedError
    if iou_type != "keypoints":
        coco_eval.params.maxDets = max_dets_per_image

    if img_ids is not None:
        coco_eval.params.imgIds = img_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


def evaluate_predictions_on_lvis(lvis_gt, lvis_results, iou_type, max_dets_per_image=None, class_names=None):
    """
    Modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/lvis_evaluation.py.
    Args:
        iou_type (str):
        max_dets_per_image (None or int): limit on maximum detections per image in evaluating AP
            This limit, by default of the LVIS dataset, is 300.
        class_names (None or list[str]): if provided, will use it to predict
            per-category AP.

    Returns:
        a dict of {metric name: score}
    """
    metrics = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
    }[iou_type]

    if len(lvis_results) == 0:
        return {metric: float("nan") for metric in metrics}

    if iou_type == "segm":
        lvis_results = copy.deepcopy(lvis_results)

        for c in lvis_results:
            c.pop("bbox", None)

    if max_dets_per_image is None:
        max_dets_per_image = 300  # Default for LVIS dataset

    lvis_results = LVISResults(lvis_gt, lvis_results, max_dets=max_dets_per_image)
    lvis_eval = LVISEval(lvis_gt, lvis_results, iou_type)
    lvis_eval.run()
    lvis_eval.print_results()

    results = lvis_eval.get_results()
    results = {metric: float(results[metric] * 100) for metric in metrics}
    print(results)

    return results


class Clicker(object):
    """
    Modified from https://github.com/SamsungLabs/ritm_interactive_segmentation/blob/b9b44603672e15aa0be878b54fd26e7e1c5d2311/isegm/inference/clicker.py#L7.
    """

    def __init__(self, gt_mask=None, init_clicks=None, ignore_label=-1, click_indx_offset=0):
        self.click_indx_offset = click_indx_offset
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None

        self.reset_clicks()

        if init_clicks is not None:
            for click in init_clicks:
                self.add_click(click)

    def make_next_click(self, pred_mask):
        assert self.gt_mask is not None
        click = self._get_next_click(pred_mask)
        self.add_click(click)

    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]

    def _get_next_click(self, pred_mask, padding=True):
        fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
        fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)

        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), "constant")
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), "constant")

        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

        return Click(is_positive=is_positive, coords=(coords_y[0], coords_x[0]))

    def add_click(self, click):
        coords = click.coords

        click.indx = self.click_indx_offset + self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.clicks_list.append(click)
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = False

    def _remove_last_click(self):
        click = self.clicks_list.pop()
        coords = click.coords

        if click.is_positive:
            self.num_pos_clicks -= 1
        else:
            self.num_neg_clicks -= 1

        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = True

    def reset_clicks(self):
        if self.gt_mask is not None:
            # self.not_clicked_map = np.ones_like(self.gt_mask, dtype=np.bool)
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=bool)

        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

        self.clicks_list = []

    def get_state(self):
        return copy.deepcopy(self.clicks_list)

    def set_state(self, state):
        self.reset_clicks()
        for click in state:
            self.add_click(click)

    def __len__(self):
        return len(self.clicks_list)


class Click:
    """
    Modified from https://github.com/SamsungLabs/ritm_interactive_segmentation/blob/b9b44603672e15aa0be878b54fd26e7e1c5d2311/isegm/inference/clicker.py#L7.
    """

    def __init__(self, is_positive, coords, indx=None):
        self.is_positive = is_positive
        self.coords = coords
        self.indx = indx

    @property
    def coords_and_indx(self):
        return (*self.coords, self.indx)

    def copy(self, **kwargs):
        self_copy = copy.deepcopy(self)
        for k, v in kwargs.items():
            setattr(self_copy, k, v)
        return self_copy
