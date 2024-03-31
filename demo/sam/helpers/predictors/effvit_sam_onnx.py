import numpy as np
import torch

from demo.sam.helpers.utils import get_device
from deployment.sam.onnx.inference import SamDecoder, SamEncoder, preprocess

DIR = "assets/export_models/sam/onnx"
get_encoder_path = lambda model_name: f"{DIR}/{model_name}_encoder.onnx"
get_decoder_path = lambda model_name: f"{DIR}/{model_name}_decoder.onnx"


class OnnxEfficientViTSamPredictor:
    def __init__(self, model_type) -> None:
        self.device = get_device()
        self.model_type = model_type
        self.encoder_model = SamEncoder(get_encoder_path(self.model_type))
        self.decoder_model = SamDecoder(get_decoder_path(model_type))
        self.reset_image()

    def reset_image(self) -> None:
        self.is_image_set = False
        self.features = None
        self.original_size = None
        self.input_size = None

    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."

        self.reset_image()

        self.original_size = image.shape[:2]

        if self.model_type in ["l0", "l1", "l2"]:
            img = preprocess(image, img_size=512)
        elif self.model_type in ["xl0", "xl1"]:
            img = preprocess(image, img_size=1024)
        else:
            raise NotImplementedError

        self.features = self.encoder_model(img)
        self.is_image_set = True

    def predict_torch(
        self,
        im_size: tuple[int, int],
        point_coords: np.ndarray = None,
        point_labels: np.ndarray = None,
        point_expansion_axis: int = 1,
        boxes: np.ndarray = None,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          im_size (tuple[int, int]): size of the cropped image
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
                                point prompts. 1 indicates a foreground point and 0 indicates a
                                background point.
          point_expansion_axis (int): dimension to expand points along.  0 to move all
            points to the same batch.  1 to make each point a separate batch
          boxes (np.ndarray or None): A Nx4 array of boxes to the model.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C=1 as the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC, C=1 containing the model's
            predictions for the quality of each mask
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            point_coords = np.expand_dims(point_coords, axis=point_expansion_axis).astype(np.float32)
            point_labels = np.expand_dims(point_labels, axis=point_expansion_axis).astype(np.float32)

        masks, iou_predictions, _ = self.decoder_model.run(
            img_embeddings=self.features,
            origin_image_size=im_size,
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=boxes,
            return_logits=return_logits,
        )

        iou_predictions = torch.from_numpy(iou_predictions)
        return masks, iou_predictions
