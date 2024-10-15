from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.utils import create_feature_extractor

from efficientvit.apps.utils.dist import is_dist_initialized, is_master, sync_tensor

__all__ = ["InceptionScoreStatsConfig", "InceptionScoreStats"]


@dataclass
class InceptionScoreStatsConfig:
    pass


def inception_score_features_to_metric(feature, splits=10, shuffle=True, rng_seed=2020):
    assert torch.is_tensor(feature) and feature.dim() == 2
    N, _ = feature.shape
    if shuffle:
        rng = np.random.RandomState(rng_seed)
        feature = feature[rng.permutation(N), :]
    feature = feature.double()

    p = feature.softmax(dim=1)
    log_p = feature.log_softmax(dim=1)

    scores = []
    for i in range(splits):
        p_chunk = p[(i * N // splits) : ((i + 1) * N // splits), :]
        log_p_chunk = log_p[(i * N // splits) : ((i + 1) * N // splits), :]
        q_chunk = p_chunk.mean(dim=0, keepdim=True)
        kl = p_chunk * (log_p_chunk - q_chunk.log())
        kl = kl.sum(dim=1).mean().exp().item()
        scores.append(kl)

    return {
        "inception_score_mean": float(np.mean(scores)),
        "inception_score_std": float(np.std(scores)),
    }


class InceptionScoreStats:
    def __init__(self, cfg: InceptionScoreStatsConfig):
        self.cfg = cfg
        # inception model
        self.feature_extractor: FeatureExtractorInceptionV3 = create_feature_extractor(
            "inception-v3-compat", ["logits_unbiased"], cuda=True
        )
        self.device = torch.device("cuda")
        # value should be Tensor with dtype uint8 and shape (B, 3, H, W)
        self.features = []

    @torch.no_grad
    def add_data(self, batch: torch.Tensor | NDArray[np.uint8]):
        if isinstance(batch, torch.Tensor):
            if batch.dtype == torch.uint8:
                pass
            elif batch.dtype == torch.float32:
                # to simulate storing and loading generated images
                # reference: torchvision save_image
                # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
                batch = (255 * batch + 0.5).clamp(0, 255).to(torch.uint8)
            else:
                raise NotImplementedError(f"dtype {batch.dtype} is not supported")
        elif isinstance(batch, np.ndarray) and batch.dtype == np.uint8:  # (N, H, W, 3)
            batch = np.transpose(batch, axes=(0, 3, 1, 2))
            batch = torch.tensor(batch, dtype=torch.uint8)
        else:
            raise TypeError(type(batch))
        if not batch.is_cuda:
            batch = batch.to(self.device)

        feature: torch.Tensor = self.feature_extractor(batch)[0]
        self.features.append(feature.cpu())

    def compute(self):
        features = torch.cat(self.features, dim=0)
        if is_dist_initialized():
            features = sync_tensor(features.cuda(), reduce="cat").cpu()

        # only compute fid score at master
        if is_dist_initialized() and not is_master():
            return {
                "inception_score_mean": 0.0,
                "inception_score_std": 0.0,
            }

        metric = inception_score_features_to_metric(features)
        return metric
