#!/usr/bin/env python3

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from apex.contrib.transducer import TransducerJoint
from beartype import beartype
from beartype.typing import Optional

from caiman_asr_train.autotune.dtypes import amp_dtype_for, to_contig
from caiman_asr_train.core.batches import BatchFirst
from caiman_asr_train.core.packing import packed_batch_offsets
from caiman_asr_train.rnnt.loss import (
    ApexTransducerLoss,
    FusedProjAndLoss,
    LossModifiers,
    TransducerLossFn,
)
from caiman_asr_train.rnnt.model import RNNT


@beartype
@dataclass(frozen=True)
class SplitMeta:
    cells: int
    max_f_len: int


@beartype
class JointLoss(torch.nn.Module, ABC):
    """
    An abstract base class defining the combined RNN-T joint network
    + transducer loss.
    """

    @abstractmethod
    def forward(
        self,
        f: BatchFirst,
        g: BatchFirst,
        t: BatchFirst,
        loss_mods: LossModifiers,
        meta: Optional[SplitMeta] = None,
    ) -> torch.Tensor: ...


@beartype
class UnfusedJointLoss(JointLoss):
    """
    A standard joint loss implementation that computes
    the joint network logits and transducer loss sequentially.
    """

    def __init__(self, model: RNNT, loss_fn: ApexTransducerLoss):
        super().__init__()
        self.joint_net = model.joint_net
        self._joint = model.joint
        self.loss_fn = loss_fn

    def forward(
        self,
        f: BatchFirst,
        g: BatchFirst,
        t: BatchFirst,
        loss_mods: LossModifiers,
        meta: Optional[SplitMeta] = None,
    ) -> torch.Tensor:
        logits, batch_offset = self._joint(
            f.contiguous(), g.contiguous(), None if meta is None else meta.cells
        )

        return self.loss_fn(
            logits,
            f.lens,
            t.data.contiguous(),
            t.lens,
            batch_offset=batch_offset,
            max_f_len=f.lens.max().item() if meta is None else meta.max_f_len,
            loss_mods=loss_mods,
        )


@beartype
class FusedJointLoss(JointLoss):
    """
    An I/O optimized joint loss that fuses the joint network
    projection and transducer loss.
    """

    def __init__(
        self,
        model: RNNT,
        loss_fn: FusedProjAndLoss,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.joint_net = model.joint_net
        self.apex_joint = TransducerJoint(
            pack_output=True, relu=True, dropout=True, dropout_prob=dropout_prob
        )
        self.proj_and_loss = loss_fn

    def forward(
        self,
        f: BatchFirst,
        g: BatchFirst,
        t: BatchFirst,
        loss_mods: LossModifiers,
        meta: Optional[SplitMeta] = None,
    ) -> torch.Tensor:
        f_len, y_len = f.lens.int(), t.lens.int()
        g_len = y_len + 1

        # This module subsumes the joint's projection, so it -- not autocast -- picks the
        # precision. f/g arrive fp32 either way, from detach_to_bf.
        dtype = amp_dtype_for(f.data)
        assert dtype in (
            torch.float16,
            torch.float32,
        ), f"the fused joint+loss only support fp16 and fp32, got {dtype}."

        f_data = to_contig(f.data, dtype)
        g_data = to_contig(g.data, dtype)

        boff = packed_batch_offsets(f_len, g_len)
        total = int(boff[-1].item()) if meta is None else meta.cells

        A = self.apex_joint(
            f_data,
            g_data,
            f_len,
            g_len,
            batch_offset=boff,
            packed_batch=total,
        )

        W, b = self.joint_net[-1].weight.t(), self.joint_net[-1].bias

        return self.proj_and_loss(
            A, W, b, t.data.contiguous(), f_len, y_len, boff, total, loss_mods
        ).mean()


@beartype
def build_joint_loss(
    model: RNNT,
    loss_fn: TransducerLossFn,
    dropout_prob: float = 0.0,
) -> JointLoss:
    if isinstance(loss_fn, FusedProjAndLoss):
        return FusedJointLoss(model, loss_fn, dropout_prob)

    return UnfusedJointLoss(model, loss_fn)
