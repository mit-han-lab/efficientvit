import os
import pathlib
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder

__all__ = ["load_image", "load_image_from_dir", "DMCrop", "CustomImageFolder", "ImageDataset"]


def load_image(data_path: str, mode="rgb") -> Image.Image:
    img = Image.open(data_path)
    if mode == "rgb":
        img = img.convert("RGB")
    return img


def load_image_from_dir(
    dir_path: str,
    suffix: str | tuple[str, ...] | list[str] = (".jpg", ".JPEG", ".png"),
    return_mode="path",
    k: Optional[int] = None,
    shuffle_func: Optional[Callable] = None,
) -> list | tuple[list, list]:
    suffix = [suffix] if isinstance(suffix, str) else suffix

    file_list = []
    for dirpath, _, fnames in os.walk(dir_path):
        for fname in fnames:
            if pathlib.Path(fname).suffix not in suffix:
                continue
            image_path = os.path.join(dirpath, fname)
            file_list.append(image_path)

    if shuffle_func is not None and k is not None:
        shuffle_file_list = shuffle_func(file_list)
        file_list = shuffle_file_list or file_list
        file_list = file_list[:k]

    file_list = sorted(file_list)

    if return_mode == "path":
        return file_list
    else:
        files = []
        path_list = []
        for file_path in file_list:
            try:
                files.append(load_image(file_path))
                path_list.append(file_path)
            except Exception:
                print(f"Fail to load {file_path}")
        if return_mode == "image":
            return files
        else:
            return path_list, files


class DMCrop:
    """center/random crop used in diffusion models"""

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, pil_image: Image.Image) -> Image.Image:
        """
        Center cropping implementation from ADM.
        https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
        """
        image_size = self.size
        if pil_image.size == (image_size, image_size):
            return pil_image

        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


class CustomImageFolder(ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable] = None, return_dict: bool = False):
        root = os.path.expanduser(root)
        self.return_dict = return_dict
        super().__init__(root, transform)

    def __getitem__(self, index: int) -> dict[str, Any] | tuple[Any, Any]:
        path, target = self.samples[index]
        image = load_image(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.return_dict:
            return {
                "index": index,
                "image_path": path,
                "image": image,
                "label": target,
            }
        else:
            return image, target


class ImageDataset(Dataset):
    def __init__(
        self,
        data_dirs: str | list[str],
        splits: Optional[str | list[Optional[str]]] = None,
        transform: Optional[Callable] = None,
        suffix=(".jpg", ".JPEG", ".png"),
        pil=True,
        return_dict=True,
    ) -> None:
        super().__init__()

        self.data_dirs = [data_dirs] if isinstance(data_dirs, str) else data_dirs
        if isinstance(splits, list):
            assert len(splits) == len(self.data_dirs)
            self.splits = splits
        elif isinstance(splits, str):
            assert len(self.data_dirs) == 1
            self.splits = [splits]
        else:
            self.splits = [None for _ in range(len(self.data_dirs))]

        self.transform = transform
        self.pil = pil
        self.return_dict = return_dict

        # load all images [image_path]
        self.samples = []
        for data_dir, split in zip(self.data_dirs, self.splits):
            if split is None:
                samples = load_image_from_dir(data_dir, suffix, return_mode="path")
            else:
                samples = []
                with open(split, "r") as fin:
                    for line in fin.readlines():
                        relative_path = line[:-1]
                        full_path = os.path.join(data_dir, relative_path)
                        samples.append(full_path)
            self.samples += samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int, skip_image=False) -> dict[str, Any]:
        image_path = self.samples[index]

        if skip_image:
            image = None
        else:
            try:
                image = load_image(image_path, return_pil=self.pil)
            except Exception:
                print(f"Fail to load {image_path}")
                raise OSError
            if self.transform is not None:
                image = self.transform(image)
        if self.return_dict:
            return {
                "index": index,
                "image_path": image_path,
                "image_name": os.path.basename(image_path),
                "data": image,
            }
        else:
            return image
