import os
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from efficientvit.apps.metrics.fid.inception import InceptionV3
from efficientvit.apps.utils.dist import get_dist_local_rank, is_dist_initialized, is_master, sync_tensor

__all__ = ["FIDStatsConfig", "FIDStats"]


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        warnings.warn(f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


@dataclass
class FIDStatsConfig:
    save_path: Optional[str] = None
    ref_path: Optional[str] = None


class FIDStats:
    def __init__(self, cfg: FIDStatsConfig):
        self.cfg = cfg
        if cfg.ref_path is not None:
            assert os.path.exists(cfg.ref_path)
        # inception model
        self.model = InceptionV3().cuda().eval()
        self.device = torch.device("cuda")
        if is_dist_initialized():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[get_dist_local_rank()],
                static_graph=True,
            )
        # value should be floats within [0, 1], on gpu
        self.transform = transforms.Normalize(mean=0.5, std=0.5, inplace=True)
        self.num_samples = 0
        self.pred_sum = None
        self.pred_dot_product_sum = None

    @torch.no_grad
    def add_data(self, batch: torch.Tensor):
        assert batch.is_cuda  # batch data should be on gpu
        if batch.dtype == torch.uint8:
            batch = batch / 255
        else:
            # to simulate storing and loading generated images
            # reference: torchvision save_image
            # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
            batch_quantized = (255 * batch + 0.5).clamp(0, 255).to(torch.uint8)
            batch = batch_quantized / 255
        batch: torch.Tensor = self.transform(batch)

        if list(batch.shape[-2:]) != [299, 299]:
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)

        pred = self.model(batch)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        self.num_samples += pred.shape[0]
        if self.pred_sum is None:
            self.pred_sum = np.zeros(pred.shape[1])
        if self.pred_dot_product_sum is None:
            self.pred_dot_product_sum = np.zeros((pred.shape[1], pred.shape[1]))
        self.pred_sum += pred.sum(axis=0)
        self.pred_dot_product_sum += pred.T @ pred

    def get_stats(self):
        num_samples, pred_sum, pred_dot_product_sum = self.num_samples, self.pred_sum, self.pred_dot_product_sum
        if is_dist_initialized():
            num_samples = sync_tensor(torch.tensor(num_samples).cuda(), reduce="sum").cpu().numpy()
            pred_sum = sync_tensor(torch.from_numpy(pred_sum).cuda(), reduce="sum").cpu().numpy()
            pred_dot_product_sum = (
                sync_tensor(torch.from_numpy(pred_dot_product_sum).cuda(), reduce="sum").cpu().numpy()
            )
            if not is_master():
                return None, None
        mu = pred_sum / num_samples
        sigma = (pred_dot_product_sum - num_samples * (mu[:, None] @ mu[None])) / (num_samples - 1)

        if self.cfg.save_path is not None:
            os.makedirs(os.path.dirname(self.cfg.save_path), exist_ok=True)
            np.savez(self.cfg.save_path, mu=mu, sigma=sigma)
        return mu, sigma

    def compute_fid(self):
        mu1, sigma1 = self.get_stats()  # every node must enter get_stats

        # only compute fid score at master
        if not is_master():
            return 0

        ref_data = np.load(self.cfg.ref_path)
        mu2, sigma2 = ref_data["mu"], ref_data["sigma"]
        fid = frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid

    def reset(self):
        self.num_samples = 0
        self.pred_sum = None
        self.pred_dot_product_sum = None
