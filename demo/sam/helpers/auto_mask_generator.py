from typing import List, Tuple

import numpy as np
import torch
from segment_anything.utils.amg import (
    MaskData,
    batched_mask_to_box,
    calculate_stability_score,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    uncrop_masks,
)

from demo.sam.helpers.predictors.effvit_sam_onnx import OnnxEfficientViTSamPredictor
from demo.sam.helpers.predictors.effvit_sam_tensorrt import TRTEfficientViTSamPredictor
from demo.sam.helpers.utils import ONNX, TENSORRT
from efficientvit.models.efficientvit.sam import EfficientViTSam, EfficientViTSamAutomaticMaskGenerator


class DemoEfficientViTSamAutomaticMaskGenerator(EfficientViTSamAutomaticMaskGenerator):
    def __init__(self, model: EfficientViTSam = None) -> None:
        super().__init__(model)

    def set_points_per_batch(self, points_per_batch):
        self.points_per_batch = points_per_batch

    def set_pred_iou_thresh(self, pred_iou_thresh):
        self.pred_iou_thresh = pred_iou_thresh

    def set_stability_score_thresh(self, stability_score_thresh):
        self.stability_score_thresh = stability_score_thresh

    def set_box_nms_thresh(self, box_nms_thresh):
        self.box_nms_thresh = box_nms_thresh


class DemoAccelEfficientViTSamAutomaticMaskGenerator(DemoEfficientViTSamAutomaticMaskGenerator):
    def __init__(self, mode: str, model_name: str, **kwargs) -> None:
        super().__init__()

        if mode == ONNX:
            self.predictor = OnnxEfficientViTSamPredictor(model_name)

        elif mode == TENSORRT:
            encoder_engine_path = kwargs["encoder_engine_path"]
            decoder_engine_path = kwargs["decoder_engine_path"]
            self.predictor = TRTEfficientViTSamPredictor(model_name, encoder_engine_path, decoder_engine_path)

        else:
            raise NotImplementedError

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        orig_h, orig_w = orig_size

        point_labels = np.ones(points.shape[0], dtype=np.float32)
        masks, iou_preds = self.predictor.predict_torch(
            im_size=im_size, point_coords=points, point_labels=point_labels, return_logits=True
        )

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(data["masks"], 0.0, self.stability_score_offset)

        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > 0.0

        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data
