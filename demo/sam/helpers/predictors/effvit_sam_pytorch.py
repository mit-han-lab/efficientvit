import numpy as np
import torch

from efficientvit.models.efficientvit.sam import EfficientViTSam, EfficientViTSamPredictor
from efficientvit.models.utils import get_device


class PyTorchEfficientViTSamPredictor(EfficientViTSamPredictor):
    def __init__(self, sam_model: EfficientViTSam) -> None:
        super().__init__(sam_model)

    def predict(
        self,
        point_coords: np.ndarray or None = None,
        point_labels: np.ndarray or None = None,
        boxes: np.ndarray or None = None,
        mask_input: np.ndarray or None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts using the currently set image.

        Arguments:
                point_coords (np.ndarray or None): A Nx2 array of point prompts to the
                model. Each point is in (X,Y) in pixels.
                point_labels (np.ndarray or None): A length N array of labels for the
                                point prompts. 1 indicates a foreground point and 0 indicates a
                                background point.
                boxes (np.ndarray or None): Bx4 array of box prompts to the
                model. Each box is in XYXY format.
                mask_input (np.ndarray): A low resolution mask input to the model, typically
                                coming from a previous prediction iteration. Has form 1xHxW, where
                                for SAM, H=W=256.
                multimask_output (bool): If true, the model will return three masks.
                                For ambiguous input prompts (such as a single click), this will often
                                produce better masks than a single prediction. If only a single
                                mask is needed, the model's predicted quality score can be used
                                to select the best mask. For non-ambiguous prompts, such as multiple
                                input prompts, multimask_output=False can give better results.
                return_logits (bool): If true, returns un-thresholded masks logits
                instead of a binary mask.

        Returns:
                (np.ndarray): The output masks in CxHxW format, where C is the
                number of masks, and (H, W) is the original image size.
                (np.ndarray): An array of length C containing the model's
                        predictions for the quality of each mask.
                (np.ndarray): An array of shape CxHxW, where C is the number
                        of masks and H=W=256. These low resolution logits can be passed to
                a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        device = get_device(self.model)
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None

        if point_coords is not None:
            assert point_labels is not None, "point_labels must be supplied if point_coords is supplied."
            point_coords = self.apply_coords(point_coords)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

        if boxes is not None:
            reshaped_boxes = np.empty((len(boxes), 4))
            for i, box in enumerate(boxes):
                reshaped_boxes[i] = self.apply_boxes(box)

            box_torch = torch.as_tensor(reshaped_boxes, dtype=torch.float, device=device)

        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        masks = masks.detach().cpu().numpy()
        iou_predictions = iou_predictions.detach().cpu().numpy()
        low_res_masks = low_res_masks.detach().cpu().numpy()

        return masks, iou_predictions, low_res_masks
