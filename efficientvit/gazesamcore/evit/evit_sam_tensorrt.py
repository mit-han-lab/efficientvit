import numpy as np
import torch

try:
    from efficientvit.gazesamcore.evit.helpers import apply_boxes, postprocess_masks, preprocess
except Exception as e:
    print(f"Skipping tensorrt-runtime import error: {e}")
    print("If using a non-tensorrt runtime, ignore.  Otherwise, please ensure tensorrt and torch2trt are installed")
    pass

__all__ = ["TrtEvitSam"]


class TrtEvitSam:
    def __init__(self, model_type, encoder, decoder) -> None:
        self.model_type = model_type
        self.encoder = encoder
        self.decoder = decoder
        self.reset_image()

    def reset_image(self) -> None:
        self.is_image_set = False
        self.features = None
        self.original_size = None

    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> torch.Tensor:
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."

        self.reset_image()

        self.original_size = image.shape[:2]

        if self.model_type in ["l0", "l1", "l2"]:
            img = preprocess(image, 512)
        elif self.model_type in ["xl0", "xl1"]:
            img = preprocess(image, 1024)
        else:
            raise NotImplementedError

        img_embedding = self.encoder(img)
        self.features = img_embedding[0].reshape(1, 256, 64, 64)

        self.is_image_set = True

        return self.features

    def predict_torch(
        self,
        im_size: tuple[int, int],
        boxes: np.ndarray = None,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          im_size (tuple[int, int]): size of the cropped image
          boxes (np.ndarray or None): A Nx4 array of box prompts
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C=1 as the
            number of masks, and (H, W) is the original image size.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        boxes = apply_boxes(boxes, self.original_size, im_size).astype(np.float32)
        boxes = torch.from_numpy(boxes).cuda()

        inputs = (self.features, boxes)
        assert all([x.dtype == torch.float32 for x in inputs])

        low_res_masks, iou_preds = self.decoder(*inputs)

        masks = postprocess_masks(low_res_masks, self.original_size)

        if not return_logits:
            masks = masks > 0.0

        return masks, iou_preds
