import math
import random
from copy import deepcopy
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

__all__ = ["SAMDistributedSampler", "RandomHFlip", "ResizeLongestSide", "Normalize_and_Pad"]


class SAMDistributedSampler(DistributedSampler):
    """
    Modified from https://github.com/pytorch/pytorch/blob/97261be0a8f09bed9ab95d0cee82e75eebd249c3/torch/utils/data/distributed.py.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        sub_epochs_per_epoch: int = 1,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        self.sub_epoch = 0
        self.sub_epochs_per_epoch = sub_epochs_per_epoch
        self.set_sub_num_samples()

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        indices = indices[(self.sub_epoch % self.sub_epochs_per_epoch) :: self.sub_epochs_per_epoch]

        return iter(indices)

    def __len__(self) -> int:
        return self.sub_num_samples

    def set_sub_num_samples(self) -> int:
        self.sub_num_samples = self.num_samples // self.sub_epochs_per_epoch
        if self.num_samples % self.sub_epochs_per_epoch > self.sub_epoch:
            self.sub_num_samples += 1

    def set_epoch_and_sub_epoch(self, epoch: int, sub_epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
            sub_epoch (int): Sub epoch number.
        """
        self.epoch = epoch
        self.sub_epoch = sub_epoch
        self.set_sub_num_samples()


class RandomHFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, masks, points, bboxs, shape = (
            sample["image"],
            sample["masks"],
            sample["points"],
            sample["bboxs"],
            sample["shape"],
        )

        if random.random() >= self.prob:
            image = torch.flip(image, dims=[2])
            masks = torch.flip(masks, dims=[2])
            points = deepcopy(points).to(torch.float)
            bboxs = deepcopy(bboxs).to(torch.float)
            points[:, 0] = shape[-1] - points[:, 0]
            bboxs[:, 0] = shape[-1] - bboxs[:, 2] - bboxs[:, 0]

        return {"image": image, "masks": masks, "points": points, "bboxs": bboxs, "shape": shape}


class ResizeLongestSide(object):
    """
    Modified from https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/utils/transforms.py.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: torch.Tensor, original_size: tuple[int, ...]) -> torch.Tensor:
        target_size = self.get_preprocess_shape(original_size[0], original_size[1], self.target_length)
        return F.interpolate(image, target_size, mode="bilinear", align_corners=False, antialias=True)

    def apply_boxes(self, boxes: torch.Tensor, original_size: tuple[int, ...]) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_coords(self, coords: torch.Tensor, original_size: tuple[int, ...]) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(original_size[0], original_size[1], self.target_length)
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

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

    def __call__(self, sample):
        image, masks, points, bboxs, shape = (
            sample["image"],
            sample["masks"],
            sample["points"],
            sample["bboxs"],
            sample["shape"],
        )

        image = self.apply_image(image.unsqueeze(0), shape).squeeze(0)
        masks = self.apply_image(masks.unsqueeze(1), shape).squeeze(1)
        points = self.apply_coords(points, shape)
        bboxs = self.apply_boxes(bboxs, shape)

        return {"image": image, "masks": masks, "points": points, "bboxs": bboxs, "shape": shape}


class Normalize_and_Pad(object):
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length
        self.transform = transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

    def __call__(self, sample):
        image, masks, points, bboxs, shape = (
            sample["image"],
            sample["masks"],
            sample["points"],
            sample["bboxs"],
            sample["shape"],
        )

        h, w = image.shape[-2:]
        image = self.transform(image)

        padh = self.target_length - h
        padw = self.target_length - w

        image = F.pad(image.unsqueeze(0), (0, padw, 0, padh), value=0).squeeze(0)
        masks = F.pad(masks.unsqueeze(1), (0, padw, 0, padh), value=0).squeeze(1)

        return {"image": image, "masks": masks, "points": points, "bboxs": bboxs, "shape": shape}
