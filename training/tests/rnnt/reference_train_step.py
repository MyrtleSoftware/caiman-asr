#!/usr/bin/env python3

from argparse import Namespace

from beartype import beartype
from beartype.typing import Optional, Tuple, Union
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from caiman_asr_train.core.batches import TimeFirst
from caiman_asr_train.rnnt.loss import ApexTransducerLoss, LossModifiers
from caiman_asr_train.rnnt.model import RNNT, StateRNNT
from caiman_asr_train.train_utils.core import autocast_if, is_loss_invalid, sync_ctx
from caiman_asr_train.train_utils.distributed import print_once, unwrap_ddp


@beartype
def reference_train_step(
    model: Union[RNNT, DDP],
    loss_fn: ApexTransducerLoss,
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
    Whole-batch ``RNNT.forward`` -> loss -> backward.
    """
    with sync_ctx(model, is_last_microbatch):
        with autocast_if(not args.no_amp):
            if rnnt_state is None:
                enc = None
                pred = unwrap_ddp(model).sos_for(feats)
            else:
                enc = rnnt_state.enc
                pred = rnnt_state.pred

            fwd = model(
                feats=feats,
                tokens=tokens,
                prefix=prefix if args.conditional else None,
                pred_state=pred,
                enc_state=enc,
            )

            t_b = tokens.T.contiguous()

            loss = loss_fn(
                fwd.packed_logits,
                fwd.f.lens,
                t_b.data,
                t_b.lens,
                batch_offset=fwd.batch_offset,
                max_f_len=fwd.f.lens.max().item(),
                loss_mods=loss_mods,
            )
            loss /= args.grad_accumulation_batches

        loss_nan = is_loss_invalid(loss, args.num_gpus)

        if loss_nan:
            print_once("WARNING: loss is non-finite; skipping update")
        elif not args.no_amp:
            assert scaler is not None
            scaler.scale(loss).backward()
        else:
            loss.backward()

    loss_item = loss.item()
    del loss
    return loss_item, loss_nan, fwd.state
