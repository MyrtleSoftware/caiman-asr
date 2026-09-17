#!/usr/bin/env python3

import time
from contextlib import contextmanager, nullcontext

import torch
from beartype import beartype
from torch.nn.parallel import DistributedDataParallel as DDP

from caiman_asr_train.log.tb_dllogger import log


def is_loss_invalid(loss: torch.Tensor, num_gpus: int) -> bool:
    """
    Check for a non-finite (NaN or +/-inf) loss, on any GPU when distributed.

    An inf loss is just as fatal as a NaN one: backpropagating it produces
    non-finite gradients, so it must drop the batch the same way (checking
    only isnan let inf losses through to backward).

    If a single GPU was checked, and some other GPU had a bad loss but this GPU
    hadn't, then the backwards pass would be skipped only on the other GPU.
    That's a problem because the GPUs sync during the backwards pass
    (https://pytorch.org/tutorials/intermediate/ddp_tutorial.html), so the GPUs
    would get out of sync. The symptom of out-of-sync GPUs is this process
    hanging at the end of the epoch, while the other process goes on to do
    validation and then hangs. Eventually the training would time out.

    This function is based on the discussion in
    https://discuss.pytorch.org/t/skiping-training-iteration-in-ddp-setting-results-in-stalled-training/177143 # noqa: E501
    """
    if num_gpus > 1:
        loss_list = [torch.empty_like(loss) for _ in range(num_gpus)]
        # all_gather broadcasts loss to all processes.
        # loss_list receives the loss from all processes.
        torch.distributed.all_gather(loss_list, loss)
        return any([not torch.isfinite(loss).all() for loss in loss_list])
    else:
        return not torch.isfinite(loss).all().item()


def are_losses_invalid(all_finite: torch.Tensor, num_gpus: int) -> bool:
    if num_gpus > 1:
        flags = [torch.empty_like(all_finite) for _ in range(num_gpus)]
        torch.distributed.all_gather(flags, all_finite)
        all_finite = torch.stack(flags).all()

    return not all_finite.item()


@contextmanager
@beartype
def autocast_if(amp: bool, device: str = "cuda"):
    """
    Enable autocast only if condition on x is met.

    Modify `should_autocast` as needed.
    """

    with torch.amp.autocast(device) if amp else nullcontext():
        yield


def sync_ctx(module, do_sync: bool):
    """
    Context manager gating DDP gradient all-reduce on the following backward.

    The wrapped module's forward must run inside the returned context: DDP decides
    whether to all-reduce at forward time, so a forward run under ``module.no_sync()``
    accumulates gradients locally and the subsequent backward (wherever it is called)
    does not all-reduce. Returns a no-op when ``do_sync`` is True or when ``module`` is
    not a DDP instance (single-GPU / bare model), so callers can use it unconditionally.
    """
    if do_sync or not isinstance(module, DDP):
        return nullcontext()

    return module.no_sync()


@beartype
def calculate_epoch(step: int, steps_per_epoch: int):
    return 1 + (step - 1) // steps_per_epoch


@beartype
def log_end_of_epoch(epoch_start_time: float, epoch: int, epoch_utts: int) -> None:
    epoch_time = time.time() - epoch_start_time
    # log epoch
    log(
        (epoch,),
        None,
        "train_avg",
        {
            "throughput-audio-samples-per-sec": epoch_utts / epoch_time,
            "took": epoch_time,
        },
    )
