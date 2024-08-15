import torch.distributed as dist
from beartype import beartype


@beartype
def get_rank_or_zero() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


@beartype
def is_rank_zero() -> bool:
    return get_rank_or_zero() == 0


def print_once(msg):
    if is_rank_zero():
        print(msg)


def unwrap_ddp(model):
    """model could be wrapped in DistributedDataParallel"""
    return getattr(model, "module", model)
