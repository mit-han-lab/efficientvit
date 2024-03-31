import argparse
from copy import deepcopy
from typing import Any, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml

from deployment.sam.onnx.export_encoder import SamResize


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))


class SamEncoder:
    def __init__(self, model_path: str, device: str = "cpu", **kwargs):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ["CUDAExecutionProvider"]
        elif device == "cpu":
            provider = ["CPUExecutionProvider"]
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")

        print(f"loading encoder model from {model_path}...")
        self.session = ort.InferenceSession(model_path, opt, providers=provider, **kwargs)
        self.input_name = self.session.get_inputs()[0].name

    def _extract_feature(self, tensor: np.ndarray) -> np.ndarray:
        feature = self.session.run(None, {self.input_name: tensor})[0]
        return feature

    def __call__(self, img: np.array, *args: Any, **kwds: Any) -> Any:
        return self._extract_feature(img)


class SamDecoder:
    def __init__(
        self, model_path: str, device: str = "cpu", target_size: int = 1024, mask_threshold: float = 0.0, **kwargs
    ):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ["CUDAExecutionProvider"]
        elif device == "cpu":
            provider = ["CPUExecutionProvider"]
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")

        print(f"loading decoder model from {model_path}...")
        self.target_size = target_size
        self.mask_threshold = mask_threshold
        self.session = ort.InferenceSession(model_path, opt, providers=provider, **kwargs)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
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
        origin_image_size: Union[list, tuple],
        point_coords: Union[list, np.ndarray] = None,
        point_labels: Union[list, np.ndarray] = None,
        boxes: Union[list, np.ndarray] = None,
        return_logits: bool = False,
    ):
        input_size = self.get_preprocess_shape(*origin_image_size, long_side_length=self.target_size)

        if point_coords is None and point_labels is None and boxes is None:
            raise ValueError("Unable to segment, please input at least one box or point.")

        if img_embeddings.shape != (1, 256, 64, 64):
            raise ValueError("Got wrong embedding shape!")

        if point_coords is not None:
            point_coords = self.apply_coords(point_coords, origin_image_size, input_size).astype(np.float32)

            prompts, labels = point_coords, point_labels

        if boxes is not None:
            boxes = self.apply_boxes(boxes, origin_image_size, input_size).astype(np.float32)
            box_labels = np.array([[2, 3] for _ in range(boxes.shape[0])], dtype=np.float32).reshape((-1, 2))

            if point_coords is not None:
                prompts = np.concatenate([prompts, boxes], axis=1)
                labels = np.concatenate([labels, box_labels], axis=1)
            else:
                prompts, labels = boxes, box_labels

        input_dict = {"image_embeddings": img_embeddings, "point_coords": prompts, "point_labels": labels}
        low_res_masks, iou_predictions = self.session.run(None, input_dict)

        masks = mask_postprocessing(low_res_masks, origin_image_size)

        if not return_logits:
            masks = masks > self.mask_threshold
        return masks, iou_predictions, low_res_masks

    def apply_coords(self, coords, original_size, new_size):
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes, original_size, new_size):
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size, new_size)
        return boxes


def preprocess(x, img_size):
    pixel_mean = [123.675 / 255, 116.28 / 255, 103.53 / 255]
    pixel_std = [58.395 / 255, 57.12 / 255, 57.375 / 255]

    x = torch.tensor(x)
    resize_transform = SamResize(img_size)
    x = resize_transform(x).float() / 255
    x = transforms.Normalize(mean=pixel_mean, std=pixel_std)(x)

    h, w = x.shape[-2:]
    th, tw = img_size, img_size
    assert th >= h and tw >= w
    x = F.pad(x, (0, tw - w, 0, th - h), value=0).unsqueeze(0).numpy()

    return x


def resize_longest_image_size(input_image_size: torch.Tensor, longest_side: int) -> torch.Tensor:
    input_image_size = input_image_size.to(torch.float32)
    scale = longest_side / torch.max(input_image_size)
    transformed_size = scale * input_image_size
    transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
    return transformed_size


def mask_postprocessing(masks: torch.Tensor, orig_im_size: torch.Tensor) -> torch.Tensor:
    img_size = 1024
    masks = torch.tensor(masks)
    orig_im_size = torch.tensor(orig_im_size)
    masks = F.interpolate(
        masks,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )

    prepadded_size = resize_longest_image_size(orig_im_size, img_size)
    masks = masks[..., : int(prepadded_size[0]), : int(prepadded_size[1])]
    orig_im_size = orig_im_size.to(torch.int64)
    h, w = orig_im_size[0], orig_im_size[1]
    masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
    return masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model type.")
    parser.add_argument(
        "--encoder_model", type=str, required=True, help="Path to the efficientvit_sam onnx encoder model."
    )
    parser.add_argument(
        "--decoder_model", type=str, required=True, help="Path to the efficientvit_sam onnx decoder model."
    )
    parser.add_argument("--img_path", type=str, default="assets/fig/cat.jpg")
    parser.add_argument("--out_path", type=str, default="assets/demo/efficientvit_sam_demo_onnx.png")
    parser.add_argument("--mode", type=str, default="point", choices=["point", "boxes"])
    parser.add_argument("--point", type=str, default=None)
    parser.add_argument("--boxes", type=str, default=None)
    args = parser.parse_args()

    encoder = SamEncoder(model_path=args.encoder_model)
    decoder = SamDecoder(model_path=args.decoder_model)

    raw_img = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)
    origin_image_size = raw_img.shape[:2]
    if args.model in ["l0", "l1", "l2"]:
        img = preprocess(raw_img, img_size=512)
    elif args.model in ["xl0", "xl1"]:
        img = preprocess(raw_img, img_size=1024)
    else:
        raise NotImplementedError

    img_embeddings = encoder(img)

    if args.mode == "point":
        H, W, _ = raw_img.shape
        point = np.array(yaml.safe_load(args.point or f"[[[{W // 2}, {H // 2}, {1}]]]"), dtype=np.float32)
        point_coords = point[..., :2]
        point_labels = point[..., 2]
        masks, _, _ = decoder.run(
            img_embeddings=img_embeddings,
            origin_image_size=origin_image_size,
            point_coords=point_coords,
            point_labels=point_labels,
        )

        plt.figure(figsize=(10, 10))
        plt.imshow(raw_img)
        for mask in masks:
            show_mask(mask, plt.gca(), random_color=len(masks) > 1)
        show_points(point_coords, point_labels, plt.gca())
        plt.axis("off")
        plt.savefig(args.out_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
        print(f"Result saved in {args.out_path}")

    elif args.mode == "boxes":
        boxes = np.array(yaml.safe_load(args.boxes), dtype=np.float32)
        masks, _, _ = decoder.run(
            img_embeddings=img_embeddings,
            origin_image_size=origin_image_size,
            boxes=boxes,
        )
        plt.figure(figsize=(10, 10))
        plt.imshow(raw_img)
        for mask in masks:
            show_mask(mask, plt.gca(), random_color=len(masks) > 1)
        for box in boxes:
            show_box(box, plt.gca())
        plt.axis("off")
        plt.savefig(args.out_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
        print(f"Result saved in {args.out_path}")

    else:
        raise NotImplementedError
