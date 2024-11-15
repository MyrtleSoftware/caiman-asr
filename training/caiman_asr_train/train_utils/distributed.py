from time import strftime

import torch.distributed as dist
from beartype import beartype

from caiman_asr_train.utils.color_print import bold_yellow


@beartype
def get_rank_or_zero() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


@beartype
def is_rank_zero() -> bool:
    return get_rank_or_zero() == 0


def print_once(msg):
    if is_rank_zero():
        print(msg)


def warn_once(msg):
    print_once(bold_yellow(msg))


def time_print_once(msg):
    print_once(f"{strftime('%c')} {msg}")


def unwrap_ddp(model):
    """model could be wrapped in DistributedDataParallel"""
    return getattr(model, "module", model)
