from dataclasses import dataclass

import torch

from efficientvit.apps.utils.dist import sync_tensor

__all__ = ["PSNRStatsConfig", "PSNRStats"]


@dataclass
class PSNRStatsConfig:
    pass


def compute_psnr(image_ref: torch.Tensor, image_pred: torch.Tensor, max_pixel: float = 255.0) -> torch.Tensor:
    """
    image_ref : (B, 3, H, W) uint8
    image_pred: (B, 3, H, W) uint8
    """
    assert image_ref.dtype == torch.uint8 and image_pred.dtype == torch.uint8
    mse = (image_ref.float() - image_pred.float()).square().mean(dim=(1, 2, 3))
    psnr = 10 * torch.log10(max_pixel**2 / mse)
    return psnr


class PSNRStats:
    def __init__(self, cfg: PSNRStatsConfig):
        self.cfg = cfg
        self.psnr_sum, self.psnr_cnt = 0, 0

    @torch.no_grad
    def add_data(self, image_ref: torch.Tensor, image_pred: torch.Tensor):
        psnr = compute_psnr(image_ref, image_pred)
        self.psnr_sum += psnr.sum().item()
        self.psnr_cnt += psnr.shape[0]

    def compute(self):
        psnr_sum = sync_tensor(self.psnr_sum, reduce="sum")
        psnr_cnt = sync_tensor(self.psnr_cnt, reduce="sum")

        if isinstance(psnr_sum, torch.Tensor):
            psnr_sum = psnr_sum.item()
        if isinstance(psnr_cnt, torch.Tensor):
            psnr_cnt = psnr_cnt.item()

        return psnr_sum / psnr_cnt

    def reset(self):
        self.psnr_sum, self.psnr_cnt = 0, 0
