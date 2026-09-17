import gc
import os

import torch
from beartype import beartype
from beartype.typing import Optional
from lhotse.dataset import (
    IterableDatasetWrapper,
    make_worker_init_fn,
)
from lhotse.dataset.dataloading import resolve_seed
from lhotse.utils import fix_random_seed

from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.lhotse.config import DataConfig, MultilingualSharManifests
from caiman_asr_train.lhotse.sampler import get_lhotse_sampler_from_config
from caiman_asr_train.lhotse.stats import EpochStats
from caiman_asr_train.lhotse.warn import lhotse_warn


class _CollectBeforeForkLoader(torch.utils.data.DataLoader):
    """
    A DataLoader that collects this process's garbage before it forks workers.

    Triton (<v3.2.0) kernel compilation leaves an unreachable MLIRContext cycle in
    this process's heap. Its destructor joins the LLVM thread pool's threads, which
    a fork does not clone, so whichever process finalises it after a fork segfaults.
    Collecting here destroys it in the parent, where those threads still exist,
    leaving nothing hazardous for the workers to inherit.
    """

    def __iter__(self):
        if self.num_workers > 0:
            gc.collect()
        return super().__iter__()


@beartype
def get_lhotse_dataloader(
    config: DataConfig,
    global_rank: int,
    world_size: int,
    dataset: torch.utils.data.Dataset,
    tokenizer: Optional[Tokenizer],
) -> tuple[torch.utils.data.DataLoader, EpochStats]:
    """
    Set up a Lhotse training dataloder.

    The ``dataset`` parameter should be an instance of a Lhotse-compatible
    PyTorch Dataset class. It only needs to define the following method
    ``__getitem__(self, cuts: CutSet) -> Any``. This dataset is not expected to
    hold a reference to any actual data; it may be interpreted as a function
    mapping a Lhotse CutSet into a mini-batch of tensors The dataset is
    expected to be stateless i.e. __getitem__ is pure in the functional sense.
    """

    if config.use_cuda:
        _maybe_set_cuda_expandable_segments(config)

    # First, resolve the random seed in case a string value was provided.
    prime_seed = resolve_seed(config.prime_seed)

    fix_random_seed(prime_seed)

    sampler, stats = get_lhotse_sampler_from_config(
        config=config,
        global_rank=global_rank,
        world_size=world_size,
        tokenizer=tokenizer,
    )

    if (
        config.warn
        and isinstance(config.manifests, MultilingualSharManifests)
        and not config.use_iterable_dataset
    ):
        lhotse_warn(
            """\
            You have requested a mappable dataset with Lhotse Shar manifests. This
            will trigger IO in the main (GPU) process, probably impacting training
            speed.\
            """
        )

    if config.use_iterable_dataset:
        # Wrapper here is recommended when using Lhotse Shar data, because then
        # I/O happens upon sampler iteration. Normally, the sampler resides in
        # the training loop process, but when we use iterable dataset, we can
        # move it to the dataloading worker process. We use lhotse's own
        # worker_init_fn which leverages information such as rank, world_size,
        # worker_id, etc. to set a different random seed for each (node,
        # worker) combination. This together with infinite datasets removes the
        # need to split data across nodes/workers.
        dloader = _CollectBeforeForkLoader(
            dataset=IterableDatasetWrapper(dataset=dataset, sampler=sampler),
            worker_init_fn=make_worker_init_fn(
                rank=global_rank,
                world_size=world_size,
                seed=prime_seed,
            ),
            persistent_workers=config.loader_workers > 0,  # Shar maintain shuffle state
            # Common
            batch_size=None,
            num_workers=config.loader_workers,
            pin_memory=config.loader_pin_memory_bool,
            prefetch_factor=config.loader_prefetch_factor,
        )

    else:
        # For non-tarred data, the sampler resides in the training loop process
        # and reads only light-weight JSON objects; it samples mini-batches and
        # passes the meta-data to Dataset, which performs the actual I/O inside
        # its __getitem__ method.
        dloader = _CollectBeforeForkLoader(
            dataset=dataset,
            sampler=sampler,
            # Common
            batch_size=None,
            num_workers=config.loader_workers,
            pin_memory=config.loader_pin_memory_bool,
            prefetch_factor=config.loader_prefetch_factor,
        )

    return dloader, stats


@beartype
def _maybe_set_cuda_expandable_segments(config: DataConfig):
    """
    Configures PyTorch memory allocator to expand existing allocated segments
    instead of re-allocating them when tensor shape grows. This can help speed
    up the training when sequence length and/or batch size change often, and
    makes GPU more robust towards OOM.

    See here for more details:
    pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
    """
    if config.cuda_expandable_segments and torch.cuda.is_available():
        if (
            config.warn
            and (value := os.environ.get("PYTORCH_CUDA_ALLOC_CONF")) is not None
            and len(value) > 0
            and "expandable_segments:True" not in value
        ):
            lhotse_warn(
                """\
                You have set PYTORCH_CUDA_ALLOC_CONF without expandable_segments:
                True option. We're setting that option anyway. To disable it, set
                cuda_expandable_segments False in dataloader configuration.\
                """
            )

        try:
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        except RuntimeError:
            if config.warn:
                lhotse_warn(
                    """\
                    Failed to set expandable_segments:True for PyTorch CUDA
                    allocator. You may get training speed improvements if you
                    enable this.\
                    """
                )
