#!/usr/bin/env python3

from argparse import Namespace

import torch
from beartype.typing import Optional, Tuple

from caiman_asr_train.rnnt.loss import ApexTransducerLoss, get_packing_meta_data
from caiman_asr_train.rnnt.model import RNNT
from caiman_asr_train.rnnt.state import RNNTState
from caiman_asr_train.train_utils.distributed import unwrap_ddp


def model_loss_forward(
    model: RNNT,
    loss_fn: ApexTransducerLoss,
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
    # note : more misleading variable names : 'log_prob*' are actually logits
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
    del log_probs, log_prob_lens

    return loss, new_rnnt_state


def model_loss_forward_train(
    model: RNNT,
    loss_fn: ApexTransducerLoss,
    args: Namespace,
    feats: torch.Tensor,
    feat_lens: torch.Tensor,
    txt: torch.Tensor,
    txt_lens: torch.Tensor,
    rnnt_state: Optional[RNNTState],
) -> Tuple[torch.Tensor, Optional[RNNTState]]:
    """ """
    loss, new_rnnt_state = model_loss_forward(
        model, loss_fn, feats, feat_lens, txt, txt_lens, rnnt_state
    )
    loss /= args.grad_accumulation_batches
    return loss, new_rnnt_state


@torch.no_grad()
def model_loss_forward_val(
    model: RNNT,
    loss_fn: ApexTransducerLoss,
    feats: torch.Tensor,
    feat_lens: torch.Tensor,
    txt: torch.Tensor,
    txt_lens: torch.Tensor,
):
    loss, _ = model_loss_forward(
        model, loss_fn, feats, feat_lens, txt, txt_lens, rnnt_state=None
    )
    return loss
