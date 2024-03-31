from copy import deepcopy

import numpy as np
import torch

from efficientvit.models.utils import get_device

try:
    import tensorrt as trt
    from torch2trt import TRTModule

    from deployment.sam.tensorrt.inference import SamResize, apply_boxes, apply_coords, mask_postprocessing, preprocess
except Exception as e:
    print(f"Skipping tensorrt-runtime import error: {e}")
    print("If using a non-tensorrt runtime, ignore.  Otherwise, please ensure tensorrt and torch2trt are installed")
    pass


def get_encoder_engine(encoder_engine_path):
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(encoder_engine_path, "rb") as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    return TRTModule(engine, input_names=["input_image"], output_names=["image_embeddings"])


def get_decoder_engine(decoder_engine_path):
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(decoder_engine_path, "rb") as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    return TRTModule(
        engine,
        input_names=["image_embeddings", "point_coords", "point_labels"],
        output_names=["masks", "iou_predictions"],
    )


class TRTEfficientViTSamPredictor:
    def __init__(self, model_type, encoder_engine_path, decoder_engine_path) -> None:
        self.model_type = model_type
        self.encoder = get_encoder_engine(encoder_engine_path)
        self.decoder = get_decoder_engine(decoder_engine_path)
        self.reset_image()

    @property
    def device(self):
        return get_device(self.model)

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
            img = preprocess(image, 512, "cuda")
        elif self.model_type in ["xl0", "xl1"]:
            img = preprocess(image, 1024, "cuda")
        else:
            raise NotImplementedError

        self.input_size = SamResize.get_preprocess_shape(*self.original_size, long_side_length=1024)
        img_embedding = self.encoder(img)
        self.features = img_embedding[0].reshape(1, 256, 64, 64)

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
          point_expansion_axis (int): dim across which to expand points.  1 to place each
            point in its own batch, 0 to place all points in the same batch
          boxes (np.ndarray or None): A Nx4 array of box prompts
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
            point_coords = np.expand_dims(point_coords, axis=point_expansion_axis)
            point_coords = apply_coords(point_coords, self.original_size, im_size).astype(np.float32)
            point_labels = np.expand_dims(point_labels, axis=point_expansion_axis).astype(np.float32)

            prompts, labels = point_coords, point_labels

        if boxes is not None:
            boxes = boxes.reshape((len(boxes), 1, 4))
            boxes = apply_boxes(boxes, self.original_size, im_size).astype(np.float32)
            box_labels = np.array([[2, 3] for _ in range(boxes.shape[0])], dtype=np.float32).reshape((-1, 2))

            if point_coords is not None:
                prompts = np.concatenate([prompts, boxes], axis=1)
                labels = np.concatenate([labels, box_labels], axis=1)
            else:
                prompts, labels = boxes, box_labels

        prompts = torch.from_numpy(prompts).cuda()
        labels = torch.from_numpy(labels).cuda()

        inputs = (self.features, prompts, labels)
        assert all([x.dtype == torch.float32 for x in inputs])

        low_res_masks, iou_predictions = self.decoder(*inputs)
        low_res_masks = low_res_masks.reshape(-1, 1, 256, 256)

        masks = mask_postprocessing(low_res_masks, self.original_size)

        if not return_logits:
            masks = masks > 0.0

        return masks, iou_predictions
