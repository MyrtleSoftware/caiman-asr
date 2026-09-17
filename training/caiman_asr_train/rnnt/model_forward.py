#!/usr/bin/env python3

import torch
from beartype import beartype
from beartype.typing import Optional

from caiman_asr_train.core.batches import TimeFirst
from caiman_asr_train.rnnt.joint_loss import JointLoss
from caiman_asr_train.rnnt.loss import LossModifiers
from caiman_asr_train.rnnt.model import RNNT
from caiman_asr_train.train_utils.batch_splitting import get_bsplit


@torch.no_grad()
@beartype
def val_loss(
    model: RNNT,
    joint_loss: JointLoss,
    feats: TimeFirst,
    tokens: TimeFirst,
    loss_mods: LossModifiers,
    prefix: Optional[TimeFirst],
    joint_lattice_cap: Optional[int] = None,
) -> torch.Tensor:
    """
    Validation loss: the forward half of ``train_step``.
    """
    enc_prefix = prefix if model.config.prefix_token_map is not None else None

    f, _ = model.encode(state=None, prefix=enc_prefix, feats=feats)
    g, _ = model.predict(state=model.sos_for(feats), prefix=prefix, tokens=tokens)

    f_b, g_b, t_b = f.T, g.T, tokens.T

    if joint_lattice_cap is None:
        return joint_loss(f_b, g_b, t_b, loss_mods)

    cells = [fl * gl for fl, gl in zip(f_b.lens.tolist(), g_b.lens.tolist())]
    bsplit = get_bsplit(cells, joint_lattice_cap)

    if len(bsplit) == 1:
        return joint_loss(f_b, g_b, t_b, loss_mods)

    batch_size = len(cells)
    total = torch.zeros((), dtype=torch.float32, device=f_b.data.device)

    for lo, hi in bsplit:
        part = joint_loss(
            f_b.sub_batch(lo, hi),
            g_b.sub_batch(lo, hi),
            t_b.sub_batch(lo, hi),
            loss_mods,
        )
        total += part.float() * ((hi - lo) / batch_size)

    return total
