#!/usr/bin/env python3
# Copyright (c) 2023, Myrtle Software Limited, www.myrtle.ai. All rights reserved.

from argparse import Namespace

import torch
from beartype.typing import Callable, Optional, Tuple
from torch.cuda.amp import GradScaler

from rnnt_train.common.helpers import print_once, unwrap_ddp
from rnnt_train.rnnt.loss import apexTransducerLoss, get_packing_meta_data
from rnnt_train.rnnt.model import RNNT
from rnnt_train.rnnt.state import RNNTState


def is_loss_nan(loss: torch.Tensor, num_gpus: int) -> bool:
    """When doing distributed training, we check if there is a NaN on any of the
    GPUs.

    If we only checked this GPU, and some other GPU had a NaN but this GPU
    didn't, then we'd skip the backwards pass only on the other GPU. That's a
    problem because the GPUs sync during the backwards pass
    (https://pytorch.org/tutorials/intermediate/ddp_tutorial.html), so the GPUs
    would get out of sync. The symptom of out-of-sync GPUs is this process
    hanging at the end of the epoch, while the other process goes on to do
    validation and then hangs. Eventually the training would time out.

    This function is based on the discussion in
    https://discuss.pytorch.org/t/skiping-training-iteration-in-ddp-setting-results-in-stalled-training/177143
    """
    if num_gpus > 1:
        loss_list = [torch.tensor(0.0).cuda() for _ in range(num_gpus)]
        # all_gather broadcasts loss to all processes.
        # loss_list receives the loss from all processes.
        torch.distributed.all_gather(loss_list, loss)
        return any([torch.isnan(l).any() for l in loss_list])
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
    model: RNNT,
    loss_fn: apexTransducerLoss,
    args: Namespace,
    feats: torch.Tensor,
    feat_lens: torch.Tensor,
    txt: torch.Tensor,
    txt_lens: torch.Tensor,
    scaler: Optional[GradScaler],
    rnnt_state=Optional[RNNTState],
) -> Tuple[float, bool, Optional[RNNTState]]:
    """
    Run a step of training.
    """
    model_forward_fn = maybe_autocast(model_forward, args.amp)
    loss, new_rnnt_state = model_forward_fn(
        model, loss_fn, args, feats, feat_lens, txt, txt_lens, rnnt_state
    )
    loss_nan = is_loss_nan(loss, args.num_gpus)
    if loss_nan:
        print_once(f"WARNING: loss is NaN; skipping update")
    else:
        if args.amp:
            # scale losses w/ pytorch AMP to prevent underflowing before backward pass
            scaler.scale(loss).backward()
        else:
            loss.backward()
    loss_item = loss.item()
    del loss
    return loss_item, loss_nan, new_rnnt_state


def model_forward(
    model: RNNT,
    loss_fn: apexTransducerLoss,
    args: Namespace,
    feats: torch.Tensor,
    feat_lens: torch.Tensor,
    txt: torch.Tensor,
    txt_lens: torch.Tensor,
    rnnt_state: Optional[RNNTState],
) -> Tuple[torch.Tensor, Optional[RNNTState]]:
    """
    Run model and get loss
    """
    # calc parameters used for the APEX transducer loss and joint implementations
    meta_data = get_packing_meta_data(
        feat_lens=feat_lens,
        txt_lens=txt_lens,
        enc_time_reduction=unwrap_ddp(model).enc_stack_time_factor,
    )
    # note : more misleading variable names : 'log_prob*' are actually logits - rob@myrtle
    log_probs, log_prob_lens, new_rnnt_state = model(
        feats,
        feat_lens,
        txt,
        txt_lens,
        batch_offset=meta_data["batch_offset"],
        enc_state=rnnt_state.enc_state if rnnt_state else None,
        pred_net_state=rnnt_state.pred_net_state if rnnt_state else None,
    )
    loss = loss_fn(
        log_probs,
        log_prob_lens,
        txt,
        txt_lens,
        meta_data["batch_offset"],
        meta_data["max_f_len"],
    )
    loss /= args.grad_accumulation_batches

    del log_probs, log_prob_lens
    return loss, new_rnnt_state
