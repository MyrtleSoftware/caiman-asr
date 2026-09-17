#!/usr/bin/env python3
from argparse import Namespace

import torch
from beartype import beartype
from beartype.typing import Optional, Tuple
from torch.amp.grad_scaler import GradScaler

from caiman_asr_train.core.batches import BatchFirst, TimeFirst
from caiman_asr_train.rnnt.joint_loss import SplitMeta
from caiman_asr_train.rnnt.loss import LossModifiers
from caiman_asr_train.rnnt.model import StateRNNT
from caiman_asr_train.rnnt.sub_models import RNNTSubModels
from caiman_asr_train.train_utils.core import (
    are_losses_invalid,
    autocast_if,
    sync_ctx,
)


@beartype
def get_bsplit(cells: list[int], max_cells: int) -> list[tuple[int, int]]:
    """Produce a list of (lo, hi) indices, one per batch split,
    such that the number of cells in each split is <= max_cells."""
    assert max_cells >= 1, f"{max_cells=} must be positive"
    assert cells, "cannot split an empty batch"

    plan: list[tuple[int, int]] = []
    lo, acc = 0, 0

    for i, c in enumerate(cells):
        if acc > 0 and acc + c > max_cells:
            plan.append((lo, i))
            lo, acc = i, 0

        acc += c

    plan.append((lo, len(cells)))

    return plan


@beartype
def detach_to_bf(t: TimeFirst, requires_grad: bool = True) -> BatchFirst:
    """Detach the encoder/prediction output into a fresh batch-first fp32 leaf.

    We need these contiguous for joint/loss. Single-allocation transpose
    + dtype-cast: empty() + copy_() lets one CUDA kernel fuse the
    de-stride and (optional) fp16 -> fp32 cast.
    """
    match t.data.dtype:
        case torch.float16 | torch.bfloat16:
            out_dtype = torch.float32
        case torch.float32 | torch.float64:
            out_dtype = t.data.dtype
        case _:
            raise ValueError(f"Unsupported dtype {t.data.dtype} for detach_to_bf")

    T, B, *rest = t.data.shape
    dst = torch.empty((B, T, *rest), dtype=out_dtype, device=t.data.device)

    with torch.no_grad():
        dst.copy_(t.data.transpose(0, 1))  # cast + de-stride in one kernel

    return BatchFirst(data=dst.requires_grad_(requires_grad), lens=t.lens)


@beartype
def train_step(
    model: RNNTSubModels,
    args: Namespace,
    feats: TimeFirst,
    tokens: TimeFirst,
    scaler: Optional[GradScaler],
    rnnt_state: Optional[StateRNNT],
    loss_mods: LossModifiers,
    prefix: Optional[TimeFirst],
    is_last_microbatch: bool,
) -> Tuple[float, bool, Optional[StateRNNT]]:
    """
    Run a step of training.

    The encoder and prediction batches are split into sub-batches, each containing
    up to ``joint_lattice_cap`` lattice cells, and the joint network plus loss is
    run on each sub-batch.

    We use a RNNTSubModels object to access the sub-models of the underlying RNNT
    class. This is necessary instead of using the RNNT class directly because, when
    running distributed training, DistributedDataParallel (DDP) requires that a
    model's forward method is called in order to correctly sync the accumulated
    gradients across all GPUs.

    The encoder, prediction and joint+loss are wrapped in separate DDPs over disjoint
    parameter sets. ``is_last_microbatch`` is True on the final gradient-accumulation
    micro-batch; each DDP uses ``no_sync`` until then (and the joint until its last
    split), so every parameter is all-reduced exactly once per optimizer step.
    """
    with autocast_if(not args.no_amp):
        if rnnt_state is None:
            enc = None
            pred = model.rnnt.sos_for(feats)
        else:
            enc = rnnt_state.enc
            pred = rnnt_state.pred

        enc_prefix = prefix if model.rnnt.config.prefix_token_map is not None else None

        with sync_ctx(model.encoder, is_last_microbatch):
            f, enc = model.encoder(state=enc, prefix=enc_prefix, feats=feats)

        with sync_ctx(model.prediction, is_last_microbatch):
            g, pred = model.prediction(state=pred, prefix=prefix, tokens=tokens)

        if enc is not None and pred is not None:
            rnnt_state = StateRNNT(enc=enc, pred=pred)
        else:
            rnnt_state = None

    # Values across the cut between (encoder, prediction) and (joint, loss): f_2/g_2
    # are those outputs as plain data with no autograd history, so a split's backward
    # stops there instead of entering the encoder. fp32 improves stability.
    f_2 = detach_to_bf(f, requires_grad=False)
    g_2 = detach_to_bf(g, requires_grad=False)
    f_grad_buf = torch.empty_like(f_2.data)
    g_grad_buf = torch.empty_like(g_2.data)

    # Targets requires contiguous for loss
    t_b: BatchFirst = tokens.T.contiguous()

    batch_size = f.lens.size(0)

    loss_acc = torch.zeros((), dtype=torch.float32, device=f_2.data.device)
    all_finite = torch.ones((), dtype=torch.bool, device=f_2.data.device)

    f_lens = f_2.lens.tolist()
    cells = [fl * gl for fl, gl in zip(f_lens, g_2.lens.tolist())]
    bsplit = get_bsplit(cells, args.joint_lattice_cap)

    for n, (lo, hi) in enumerate(bsplit):
        is_last_split = n + 1 == len(bsplit)
        loss_weight = (hi - lo) / (batch_size * args.grad_accumulation_batches)

        # Sub batches, use tight as we don't want to have to make contiguous again
        f_part = f_2.sub_batch(lo, hi, tight=False)
        g_part = g_2.sub_batch(lo, hi, tight=False)
        t_part = t_b.sub_batch(lo, hi, tight=False)

        # Each split is an autograd leaf
        f_part.data.requires_grad_(True)
        g_part.data.requires_grad_(True)

        # The joint and loss DDP all-reduces only on the last split of the
        # last micro-batch.
        with sync_ctx(model.joint_loss, is_last_microbatch and is_last_split):
            with autocast_if(not args.no_amp):
                meta = SplitMeta(cells=sum(cells[lo:hi]), max_f_len=max(f_lens[lo:hi]))
                loss = model.joint_loss(f_part, g_part, t_part, loss_mods, meta)
                loss = loss * loss_weight

            with torch.no_grad():
                loss_acc += loss
                all_finite &= torch.isfinite(loss).all()

            # Backward through the joint and loss per split;
            if not args.no_amp:
                # Scale losses w/ pytorch AMP to prevent underflow before backward pass
                assert scaler is not None
                scaler.scale(loss).backward()
            else:
                loss.backward()

        f_grad_buf[lo:hi].copy_(f_part.data.grad)
        g_grad_buf[lo:hi].copy_(g_part.data.grad)

        # Make sure no references are held into the next iteration
        del f_part, g_part, t_part, loss

    del t_b

    loss_item = loss_acc.item()
    batch_has_nan = are_losses_invalid(all_finite, args.num_gpus)

    # The buffers now hold dL/df and dL/dg for the micro-batch; hand them to the
    # encoder and prediction graphs.
    # It is faster to match memory format before call to backward
    f_grad = f_grad_buf.transpose(0, 1).contiguous()
    del f_2, f_grad_buf
    f.data.backward(f_grad)
    del f_grad

    g_grad = g_grad_buf.transpose(0, 1).contiguous()
    del g_2, g_grad_buf
    g.data.backward(g_grad)
    del g_grad

    return loss_item, batch_has_nan, rnnt_state
