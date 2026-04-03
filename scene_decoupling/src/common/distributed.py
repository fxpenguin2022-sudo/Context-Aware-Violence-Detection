from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class DistEnv:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int


def init_distributed(backend: str = 'nccl') -> DistEnv:
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local = int(os.environ.get('LOCAL_RANK', '0'))

    if world > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        return DistEnv(True, rank, world, local)
    return DistEnv(False, 0, 1, 0)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return get_rank() == 0


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if get_world_size() == 1:
        return x
    out = x.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


def gather_objects(obj: Any) -> list[Any]:
    if get_world_size() == 1:
        return [obj]
    out = [None for _ in range(get_world_size())]
    dist.all_gather_object(out, obj)
    return out
