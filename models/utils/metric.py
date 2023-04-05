from typing import Union

import torch

__all__ = ["AverageMeter"]


class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val: Union[torch.Tensor, int, float], delta_n=1):
        self.count += delta_n
        self.sum += val * delta_n

    def get_count(self) -> Union[torch.Tensor, int, float]:
        return (
            self.count.item()
            if isinstance(self.count, torch.Tensor) and self.count.numel() == 1
            else self.count
        )

    @property
    def avg(self):
        avg = -1 if self.count == 0 else self.sum / self.count
        return avg.item() if isinstance(avg, torch.Tensor) and avg.numel() == 1 else avg
