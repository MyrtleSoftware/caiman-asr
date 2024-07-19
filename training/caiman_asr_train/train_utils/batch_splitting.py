#!/usr/bin/env python3
from argparse import Namespace

import torch
from beartype import beartype
from beartype.typing import Optional, Tuple
from torch.cuda.amp import GradScaler

from caiman_asr_train.rnnt.loss import ApexTransducerLoss, get_packing_meta_data
from caiman_asr_train.rnnt.state import RNNTState
from caiman_asr_train.rnnt.sub_models import RNNTSubModels
from caiman_asr_train.train_utils.core import is_loss_nan, maybe_autocast
from caiman_asr_train.train_utils.distributed import print_once


def joint_and_loss(
    model: RNNTSubModels,
    loss_fn: ApexTransducerLoss,
    args: Namespace,
    f: torch.Tensor,
    f_lens: torch.Tensor,
    g: torch.Tensor,
    g_lens: torch.Tensor,
    txt: torch.Tensor,
    txt_lens: torch.Tensor,
    meta_data: dict,
) -> torch.Tensor:
    """
    Run joint and loss.
    """
    logits = model.joint(f, g, f_lens, g_lens, batch_offset=meta_data["batch_offset"])

    loss = loss_fn(
        logits,
        f_lens,
        txt,
        txt_lens,
        meta_data["batch_offset"],
        meta_data["max_f_len"],
    )

    loss /= args.grad_accumulation_batches * args.batch_split_factor

    del logits
    return loss


@beartype
def train_step_batch_split(
    model: RNNTSubModels,
    loss_fn: ApexTransducerLoss,
    args: Namespace,
    feats: torch.Tensor,
    feat_lens: torch.Tensor,
    txt: torch.Tensor,
    txt_lens: torch.Tensor,
    scaler: Optional[GradScaler],
    rnnt_state: Optional[RNNTState],
) -> Tuple[float, bool, Optional[RNNTState]]:
    """
    Run a step of training when batch_split_factor > 1.

    In this function the encoder and prediction batches are split into batch_split_factor
    sub-batches and the joint network is run on each sub-batch.

    In this case we use a RNNTSubModels object to access the sub-models of the
    underlying RNNT class. This is necessary instead of using the RNNT class directly
    because, when running distributed training, DistributedDataParallel (DDP) requires
    that a model's forward method is called in order to correctly sync the accumulated
    gradients across all GPUs.
    """
    meta_data = []
    batch_size = len(feat_lens)
    b_split = batch_size // args.batch_split_factor

    enc_pred_fn = maybe_autocast(model.rnnt.enc_pred_static, not args.no_amp)
    enc_time_stack_factor = model.rnnt.enc_stack_time_factor
    joint_fn = maybe_autocast(joint_and_loss, not args.no_amp)

    for i in range(args.batch_split_factor):
        meta_entry = get_packing_meta_data(
            feat_lens=feat_lens[i * b_split : (i + 1) * b_split],
            txt_lens=txt_lens[i * b_split : (i + 1) * b_split],
            enc_time_reduction=enc_time_stack_factor,
        )
        meta_data.append(meta_entry)

    loss_item, loss_nan = 0.0, False

    (f, f_lens), (g, g_lens), new_rnnt_state = enc_pred_fn(
        feats,
        feat_lens,
        txt,
        txt_lens,
        encode=model.encoder,
        predict=model.prediction,
        enc_state=rnnt_state.enc_state if rnnt_state else None,
        pred_net_state=rnnt_state.pred_net_state if rnnt_state else None,
    )
    # detach f and g from graph so that we don't backprop through the encoder and
    # prediction networks for each batch split element
    f_2, g_2 = f.detach(), g.detach()
    f_2.requires_grad = True
    g_2.requires_grad = True
    for i in range(args.batch_split_factor):
        loss = joint_fn(
            model=model,
            loss_fn=loss_fn,
            args=args,
            f=f_2[i * b_split : (i + 1) * b_split],
            f_lens=f_lens[i * b_split : (i + 1) * b_split],
            g=g_2[i * b_split : (i + 1) * b_split],
            g_lens=g_lens[i * b_split : (i + 1) * b_split],
            txt=txt[i * b_split : (i + 1) * b_split],
            txt_lens=txt_lens[i * b_split : (i + 1) * b_split],
            meta_data=meta_data[i],
        )

        loss_nan = is_loss_nan(loss, args.num_gpus)
        if loss_nan:
            print_once("WARNING: loss is NaN; skipping update")
            # exit this train_step early
            return loss_item, loss_nan, None

        # run backwards in joint network only and accumulate gradients in f_2 and g_2
        if not args.no_amp:
            # scale losses w/ pytorch AMP to prevent underflowing before backward pass
            scaler.scale(loss).backward()
        else:
            loss.backward()
        loss_item += loss.item()

    # We now have accumulated gradients in f_2 and g_2 which we can use to backpropagate
    # gradients through the encoder and prediction network
    f.backward(f_2.grad)
    g.backward(g_2.grad)

    return loss_item, loss_nan, new_rnnt_state
