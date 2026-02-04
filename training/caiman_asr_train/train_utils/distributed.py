import os
import shutil
from pathlib import Path
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


def barrier_if_init():
    """
    Calls torch.distributed.barrier() if torch.distributed is initialized.
    """
    if dist.is_initialized():
        dist.barrier()


@beartype
def maybe_cached(
    populate: Callable[[str | Path], None],
    cache_dir: str | Path,
):
    """
    A multi-GPU safe cache retrieval function.

    Rank 0 will execute `populate(cache_dir)` if `cache_dir` did not exist.
    `populate` is expected to create and populate `cache_dir`

    Args:
        cache_dir: Cache directory, if this exists it is assumed that the cache is valid.
        populate: A function that populates the cache directory.
    """

    if os.path.exists(cache_dir):
        print_once(f"Cache-hit@{cache_dir}")
        return

    print_once(f"Cache-miss@{cache_dir}")

    # Wait for everyone to observe the cache miss.
    barrier_if_init()

    if not dist.is_initialized() or dist.get_rank() == 0:
        try:
            populate(cache_dir)

            if not os.path.exists(cache_dir):
                raise Exception(
                    f"Cache directory `{cache_dir}` was not created by populate"
                )

        except Exception:
            # Clean up if populate fails.
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            raise
        finally:
            # Never miss the barrier.
            barrier_if_init()
    else:
        # Wait for cache to be populated.
        barrier_if_init()
