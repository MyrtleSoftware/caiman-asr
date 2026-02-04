#!/usr/bin/env python3
from argparse import Namespace

import torch
from beartype import beartype
from beartype.typing import Dict, Optional, Tuple
from torch.cuda.amp import GradScaler

from caiman_asr_train.rnnt.loss import (
    ApexTransducerLoss,
    LossModifiers,
    get_packing_meta_data,
)
from caiman_asr_train.rnnt.state import RNNTState
from caiman_asr_train.rnnt.sub_models import RNNTSubModels
from caiman_asr_train.train_utils.core import is_loss_nan, maybe_autocast


@beartype
def strip_prefix_tokens(
    txt: torch.Tensor,
    txt_lens: torch.Tensor,
    g: torch.Tensor,
    g_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Strip prefix tokens from both transcript and prediction features.

    This approach removes prefix tokens from prediction features -
    making the final prefix token the new SOS -
    and strips the corresponding tokens from the transcript for loss computation.

    Args:
        txt: Transcript tensor of shape (B, U+prefix_length)
        txt_lens: Transcript lengths of shape (B,)
        g: Prediction features of shape (B, U+prefix_length+1, H)
        g_lens: Prediction lengths of shape (B,)
        prefix_lens: Tensor of prefix lengths of shape (B,)
                     Note: All prefix lengths must be the same for now.

    Returns:
        Tuple of (stripped_txt, adjusted_txt_lens, stripped_g, adjusted_g_lens)
    """
    B = txt.shape[0]
    assert (
        prefix_lens.dim() == 1 and prefix_lens.shape[0] == B
    ), f"prefix_lens must be a 1D tensor of shape (B,), got {prefix_lens.shape}"

    # Fixed prefix length for all samples (for now)
    first = prefix_lens[0]
    assert torch.all(prefix_lens == first), "All prefix lengths must be the same"

    p = int(first.item())
    assert p >= 0, f"prefix_length must be non-negative, got {p}"

    too_short_mask = txt_lens < p
    if torch.any(too_short_mask):
        bad_idxs = torch.where(too_short_mask)[0]
        raise ValueError(
            f"Prefix length {p} exceeds transcript length for {len(bad_idxs)} samples."
        )

    # Strip prefix tokens from transcript
    stripped_txt = txt[:, p:]  # (B, U+p) ->(B, U)
    adjusted_txt_lens = txt_lens - p

    # Strip prefix tokens from prediction features
    stripped_g = g[:, p:, :]  # (B, U+p+1, H) -> (B, U+1, H)
    adjusted_g_lens = g_lens - p

    return (
        stripped_txt.contiguous(),
        adjusted_txt_lens,
        stripped_g.contiguous(),
        adjusted_g_lens,
    )


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
    meta_data: Optional[Dict],
    loss_mods: LossModifiers,
    prefix_lens: Optional[torch.Tensor] = None,
    original_feat_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Run joint and loss.
    """
    if args.conditional:
        txt, txt_lens, g, g_lens = strip_prefix_tokens(
            txt, txt_lens, g, g_lens, prefix_lens
        )

        # Recalculate metadata with adjusted transcript lengths
        # Use original feat_lens (before enc reduction) not f_lens (after enc reduction)
        if original_feat_lens is not None:
            meta_data = get_packing_meta_data(
                feat_lens=original_feat_lens,
                txt_lens=txt_lens,
                enc_time_reduction=model.rnnt.enc_stack_time_factor,
            )

    logits = model.joint(f, g, f_lens, g_lens, batch_offset=meta_data["batch_offset"])

    loss = loss_fn(
        logits,
        f_lens,
        txt,
        txt_lens,
        meta_data["batch_offset"],
        meta_data["max_f_len"],
        loss_mods=loss_mods,
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
    loss_mods: LossModifiers,
    prefix_lens: Optional[torch.Tensor] = None,
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

    if not args.conditional:
        # If conditional training, meta data is computed after stripping prefix tokens
        # so we skip computing it here
        for i in range(args.batch_split_factor):
            meta_entry = get_packing_meta_data(
                feat_lens=feat_lens[i * b_split : (i + 1) * b_split],
                txt_lens=txt_lens[i * b_split : (i + 1) * b_split],
                enc_time_reduction=enc_time_stack_factor,
            )
            meta_data.append(meta_entry)

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

    loss_item, batch_has_nan = 0.0, False

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
            meta_data=meta_data[i] if not args.conditional else None,
            loss_mods=loss_mods,
            prefix_lens=(
                prefix_lens[i * b_split : (i + 1) * b_split]
                if prefix_lens is not None
                else None
            ),
            original_feat_lens=feat_lens[i * b_split : (i + 1) * b_split],
        )

        if is_loss_nan(loss, args.num_gpus):
            batch_has_nan = True

        # Run backwards in joint network only and accumulate gradients in f_2 and g_2
        if not args.no_amp:
            # Scale losses w/ pytorch AMP to prevent underflowing before backward pass
            scaler.scale(loss).backward()
        else:
            loss.backward()

        loss_item += loss.item()

    # We now have accumulated gradients in f_2 and g_2 which we can use to backpropagate
    # gradients through the encoder and prediction network
    f.backward(f_2.grad)
    g.backward(g_2.grad)

    return loss_item, batch_has_nan, None if batch_has_nan else new_rnnt_state
