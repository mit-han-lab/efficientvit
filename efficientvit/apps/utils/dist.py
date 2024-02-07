# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import os

import torch
import torch.distributed

from efficientvit.models.utils.list import list_mean, list_sum

__all__ = [
    "dist_init",
    "get_dist_rank",
    "get_dist_size",
    "is_master",
    "dist_barrier",
    "get_dist_local_rank",
    "sync_tensor",
]


def dist_init() -> None:
    try:
        torch.distributed.init_process_group(backend="nccl")
        assert torch.distributed.is_initialized()
    except Exception:
        # use torchpack
        from torchpack import distributed as dist

        dist.init()
        os.environ["RANK"] = f"{dist.rank()}"
        os.environ["WORLD_SIZE"] = f"{dist.size()}"
        os.environ["LOCAL_RANK"] = f"{dist.local_rank()}"


def get_dist_rank() -> int:
    return int(os.environ["RANK"])


def get_dist_size() -> int:
    return int(os.environ["WORLD_SIZE"])


def is_master() -> bool:
    return get_dist_rank() == 0


def dist_barrier() -> None:
    torch.distributed.barrier()


def get_dist_local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def sync_tensor(tensor: torch.Tensor or float, reduce="mean") -> torch.Tensor or list[torch.Tensor]:
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.Tensor(1).fill_(tensor).cuda()
    tensor_list = [torch.empty_like(tensor) for _ in range(get_dist_size())]
    torch.distributed.all_gather(tensor_list, tensor.contiguous(), async_op=False)
    if reduce == "mean":
        return list_mean(tensor_list)
    elif reduce == "sum":
        return list_sum(tensor_list)
    elif reduce == "cat":
        return torch.cat(tensor_list, dim=0)
    elif reduce == "root":
        return tensor_list[0]
    else:
        return tensor_list
