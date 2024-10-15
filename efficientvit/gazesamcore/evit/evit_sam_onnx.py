import numpy as np
import onnxruntime as ort
import torch

from efficientvit.gazesamcore.evit.helpers import apply_boxes, postprocess_masks, preprocess
from efficientvit.gazesamcore.utils.timer import Timer

__all__ = ["OnnxEvitSam", "OnnxEvitSamEncoder", "OnnxEvitSamDecoder"]


class OnnxEvitSam:
    def __init__(self, model_type, encoder_model, decoder_model, device="cuda") -> None:
        self.device = device
        self.model_type = model_type
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.reset_image()

    def reset_image(self) -> None:
        self.is_image_set = False
        self.features = None
        self.original_size = None
        self.input_size = None

    def set_image(self, image: np.ndarray, image_format: str = "RGB", timer: Timer = None) -> None:
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."

        self.reset_image()

        self.original_size = image.shape[:2]

        if self.device == "cuda":
            image = torch.tensor(image).cuda()

        if self.model_type in ["l0", "l1", "l2"]:
            img = preprocess(image, img_size=512)
        elif self.model_type in ["xl0", "xl1"]:
            img = preprocess(image, img_size=1024)
        else:
            raise NotImplementedError

        if timer is not None:
            timer.start("evit encoder")
        self.features = self.encoder_model(img)
        if timer is not None:
            timer.stop("evit encoder")

        self.is_image_set = True

    def predict_torch(
        self,
        im_size: tuple[int, int],
        boxes: np.ndarray = None,
        return_logits: bool = False,
        timer: Timer = None,
    ) -> torch.Tensor:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          im_size (tuple[int, int]): size of the cropped image
          boxes (np.ndarray or None): A Nx4 array of boxes to the model.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.
          timer (Timer): Used to record model latency.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C=1 as the
            number of masks, and (H, W) is the original image size.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if timer is not None:
            timer.start("evit decoder")
        masks, iou_preds = self.decoder_model.run(
            img_embeddings=self.features,
            origin_image_size=im_size,
            boxes=boxes,
            return_logits=return_logits,
        )
        if timer is not None:
            timer.stop("evit decoder")

        masks = torch.tensor(masks, device=self.device)
        iou_preds = torch.tensor(iou_preds, device=self.device)
        return masks, iou_preds


class OnnxEvitSamEncoder:
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        self.device = device

        if device == "cuda":
            provider = ["CUDAExecutionProvider"]
        elif device == "cpu":
            provider = ["CPUExecutionProvider"]
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")

        self.session = ort.InferenceSession(model_path, providers=provider, **kwargs)
        self.input_name = self.session.get_inputs()[0].name

    def _extract_feature(self, tensor: torch.tensor) -> np.ndarray:
        io_binding = self.session.io_binding()

        tensor = tensor.contiguous()

        io_binding.bind_input(
            name="input_image",
            device_type=self.device,
            device_id=0,
            element_type=np.float32,
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )

        output = torch.empty((1, 256, 64, 64), dtype=torch.float32, device=self.device).contiguous()
        io_binding.bind_output(
            name="image_embeddings",
            device_type=self.device,
            device_id=0,
            element_type=np.float32,
            shape=tuple(output.shape),
            buffer_ptr=output.data_ptr(),
        )

        self.session.run_with_iobinding(io_binding)

        return output

    def __call__(self, img: np.array):
        return self._extract_feature(img)


class OnnxEvitSamDecoder:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        target_size: int = 1024,
        mask_threshold: float = 0.0,
        **kwargs,
    ):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ["CUDAExecutionProvider"]
        elif device == "cpu":
            provider = ["CPUExecutionProvider"]
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")

        self.target_size = target_size
        self.mask_threshold = mask_threshold
        self.session = ort.InferenceSession(model_path, opt, providers=provider, **kwargs)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def run(
        self,
        img_embeddings: np.ndarray,
        origin_image_size: list | tuple,
        boxes: list | np.ndarray = None,
        return_logits: bool = False,
    ):
        input_size = self.get_preprocess_shape(*origin_image_size, long_side_length=self.target_size)

        if img_embeddings.shape != (1, 256, 64, 64):
            raise ValueError("Got wrong embedding shape!")

        boxes = apply_boxes(boxes, origin_image_size, input_size).astype(np.float32)

        io_binding = self.session.io_binding()

        io_binding.bind_cpu_input("image_embeddings", img_embeddings.cpu().numpy())
        io_binding.bind_cpu_input("boxes", boxes)

        io_binding.bind_output("low_res_masks")
        io_binding.bind_output("iou_predictions")

        self.session.run_with_iobinding(io_binding)
        low_res_masks = io_binding.copy_outputs_to_cpu()[0]
        iou_preds = io_binding.copy_outputs_to_cpu()[1]

        masks = postprocess_masks(low_res_masks, origin_image_size)

        if not return_logits:
            masks = masks > self.mask_threshold
        return masks, iou_preds
