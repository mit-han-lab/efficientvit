import argparse
import math
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from efficientvit.models.utils import resize
from efficientvit.seg_model_zoo import create_seg_model
from eval_seg_model import ADE20KDataset, CityscapesDataset, Resize, ToTensor, get_canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="assets/fig/indoor.jpg")
    parser.add_argument("--dataset", type=str, default="ade20k", choices=["cityscapes", "ade20k"])
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--model", type=str, default="l2")
    parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="assets/demo/efficientvit_seg_demo.png")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    image = np.array(Image.open(args.image_path).convert("RGB"))
    data = image
    if args.dataset == "cityscapes":
        transform = transforms.Compose(
            [
                Resize((args.crop_size, args.crop_size * 2)),
                ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        class_colors = CityscapesDataset.class_colors
    elif args.dataset == "ade20k":
        h, w = image.shape[:2]
        if h < w:
            th = args.crop_size
            tw = math.ceil(w / h * th / 32) * 32
        else:
            tw = args.crop_size
            th = math.ceil(h / w * tw / 32) * 32
        if th != h or tw != w:
            data = cv2.resize(
                image,
                dsize=(tw, th),
                interpolation=cv2.INTER_CUBIC,
            )

        transform = transforms.Compose(
            [
                ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        class_colors = ADE20KDataset.class_colors
    else:
        raise NotImplementedError
    data = transform({"data": data, "label": np.ones_like(data)})["data"]

    model = create_seg_model(args.model, args.dataset, weight_url=args.weight_url).cuda()
    model.eval()

    with torch.inference_mode():
        data = torch.unsqueeze(data, dim=0).cuda()
        output = model(data)
        # resize the output to match the shape of the mask
        if output.shape[-2:] != image.shape[:2]:
            output = resize(output, size=image.shape[:2])
        output = torch.argmax(output, dim=1).cpu().numpy()[0]
        canvas = get_canvas(image, output, class_colors)
        canvas = Image.fromarray(canvas).save(args.output_path)


if __name__ == "__main__":
    main()
