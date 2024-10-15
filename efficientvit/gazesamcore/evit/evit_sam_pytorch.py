from typing import Optional

import numpy as np
import torch

from efficientvit.gazesamcore.utils.timer import Timer
from efficientvit.models.efficientvit.sam import EfficientViTSam, EfficientViTSamPredictor
from efficientvit.models.utils import get_device

__all__ = ["PytorchEvitSam"]


class PytorchEvitSam(EfficientViTSamPredictor):
    def __init__(self, sam_model: EfficientViTSam) -> None:
        super().__init__(sam_model)

    def predict(
        self,
        boxes: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        timer: Timer = None,
    ) -> torch.Tensor:
        """
        Predict masks for the given input prompts using the currently set image.

        Arguments:
                boxes (np.ndarray or None): Bx4 array of box prompts to the
                                model. Each box is in XYXY format.
                multimask_output (bool): If true, the model will return three masks.
                                For ambiguous input prompts (such as a single click), this will often
                                produce better masks than a single prediction. If only a single
                                mask is needed, the model's predicted quality score can be used
                                to select the best mask. For non-ambiguous prompts, such as multiple
                                input prompts, multimask_output=False can give better results.
                return_logits (bool): If true, returns un-thresholded masks logits
                                instead of a binary mask.
                timer (Timer): Used to record model latency.

        Returns:
                (np.ndarray): The output masks in CxHxW format, where C is the
                number of masks, and (H, W) is the original image size.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        device = get_device(self.model)

        reshaped_boxes = np.empty((len(boxes), 4))
        for i, box in enumerate(boxes):
            reshaped_boxes[i] = self.apply_boxes(box)

        box_torch = torch.as_tensor(reshaped_boxes, dtype=torch.float, device=device)

        if timer is not None:
            timer.start("evit decoder")

        masks, iou_preds, _ = self.predict_torch(
            None,
            None,
            box_torch,
            None,
            multimask_output,
            return_logits=return_logits,
        )

        if timer is not None:
            timer.stop("evit decoder")

        # select highest quality mask for each bbox input
        best_idx = torch.argmax(iou_preds, dim=1)
        masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
        iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)

        return masks, iou_preds
