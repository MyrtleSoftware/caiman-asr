from time import strftime

import torch.distributed as dist
from beartype import beartype
from beartype.typing import Callable, ParamSpec, TypeVar

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


def scoped_time_once(what: str):
    """
    Generic a decorator to log time of function execution to stdout.
    """
    P = ParamSpec("P")
    R = TypeVar("R")

    @beartype
    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            time_print_once(f"Starting {what}")
            result = f(*args, **kwargs)
            time_print_once(f"Finished {what}")
            return result

        return wrapper

    return decorator
