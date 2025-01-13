import os
from dataclasses import dataclass

import torch
import torchvision.transforms as transforms
from diffusers import AutoencoderDC
from omegaconf import MISSING, OmegaConf
from PIL import Image
from torchvision.utils import save_image


@dataclass
class DemoDCAEModelConfig:
    model: str = MISSING
    run_dir: str = MISSING
    input_path_list: tuple[str] = MISSING


def main():
    torch.set_grad_enabled(False)
    cfg: DemoDCAEModelConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(DemoDCAEModelConfig), OmegaConf.from_cli())
    )

    device = torch.device("cuda")
    dc_ae: AutoencoderDC = (
        AutoencoderDC.from_pretrained(f"mit-han-lab/{cfg.model}", torch_dtype=torch.float32).to(device).eval()
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )

    os.makedirs(cfg.run_dir, exist_ok=True)

    for input_path in cfg.input_path_list:
        image = Image.open(input_path)
        target_w, target_h = (
            image.size[0] // dc_ae.spatial_compression_ratio * dc_ae.spatial_compression_ratio,
            image.size[1] // dc_ae.spatial_compression_ratio * dc_ae.spatial_compression_ratio,
        )
        image = image.crop((0, 0, target_w, target_h))
        x = transform(image)[None].to(device)
        latent = dc_ae.encode(x).latent
        y = dc_ae.decode(latent).sample
        save_image(torch.cat([x, y], dim=3) * 0.5 + 0.5, os.path.join(cfg.run_dir, os.path.basename(input_path)))


if __name__ == "__main__":
    main()
