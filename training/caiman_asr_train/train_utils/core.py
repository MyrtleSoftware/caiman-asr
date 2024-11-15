#!/usr/bin/env python3

import time
from argparse import Namespace

import torch
from beartype import beartype
from beartype.typing import Callable, Optional, Tuple, Union
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from caiman_asr_train.log.tb_dllogger import log
from caiman_asr_train.rnnt.loss import ApexTransducerLoss, LossModifiers
from caiman_asr_train.rnnt.model import RNNT
from caiman_asr_train.rnnt.model_forward import model_loss_forward_train
from caiman_asr_train.rnnt.state import RNNTState
from caiman_asr_train.train_utils.distributed import print_once


def is_loss_nan(loss: torch.Tensor, num_gpus: int) -> bool:
    """When doing distributed training, check for a NaN on any of the GPUs.

    If a single GPU was checked, and some other GPU had a NaN but this GPU
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
        return any([torch.isnan(loss).any() for loss in loss_list])
    else:
        return torch.isnan(loss).any().item()


def maybe_autocast(function: Callable, amp: bool):
    """
    Optionally call function with autocast wrapper.
    """
    if amp:

        def auto_cast_wrap(*args, **kwargs):
            with torch.cuda.amp.autocast():
                return function(*args, **kwargs)

        return auto_cast_wrap
    return function


def train_step(
    model: Union[RNNT, DDP],
    loss_fn: ApexTransducerLoss,
    args: Namespace,
    feats: torch.Tensor,
    feat_lens: torch.Tensor,
    txt: torch.Tensor,
    txt_lens: torch.Tensor,
    scaler: Optional[GradScaler],
    rnnt_state: Optional[RNNTState],
    loss_mods: LossModifiers,
) -> Tuple[float, bool, Optional[RNNTState]]:
    """
    Run a step of training.
    """
    model_fwd_fn = maybe_autocast(model_loss_forward_train, not args.no_amp)
    loss, new_rnnt_state = model_fwd_fn(
        model, loss_fn, args, feats, feat_lens, txt, txt_lens, rnnt_state, loss_mods
    )
    loss_nan = is_loss_nan(loss, args.num_gpus)
    if loss_nan:
        print_once("WARNING: loss is NaN; skipping update")
    else:
        if not args.no_amp:
            # scale losses w/ pytorch AMP to prevent underflowing before backward pass
            scaler.scale(loss).backward()
        else:
            loss.backward()
    loss_item = loss.item()
    del loss
    return loss_item, loss_nan, new_rnnt_state


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
