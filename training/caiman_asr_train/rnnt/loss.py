# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, replace

import torch
from beartype import beartype
from beartype.typing import Literal, Optional, Union
from rnnt_ext.transducer.loss import TransducerLoss as CUDATransducerLoss

from caiman_asr_train.core.packing import packed_batch_offsets
from caiman_asr_train.rnnt.loss_modifiers import (  # noqa: F401
    IDENTITY_LOSS_MODIFIERS,
    LossModifiers,
)
from caiman_asr_train.rnnt.triton_loss import TritonTransducerLoss
from caiman_asr_train.tile_transducer.autotune import (
    Mode,
    TileLangKernels,
    select_tilelang_kernels,
)
from caiman_asr_train.tile_transducer.tilelang_loss import TileLangTransducerLoss
from caiman_asr_train.tile_transducer.tilelang_proj_and_loss import TileLangProjAndLoss
from caiman_asr_train.train_utils.distributed import print_once


@beartype
@dataclass(frozen=True)
class LossSpec:
    """Everything ``build_loss_fn`` needs to build one loss at one precision."""

    backend: str
    blank_idx: int
    eos_idx: Optional[int]
    star_idx: Optional[int]
    packed_input: bool
    joint_n_hid: int
    device: torch.device
    dtype: torch.dtype
    # dp_mu and dp_clamp fixed for the whole run.
    dp_mu: float = 0.0
    dp_clamp: Optional[float] = None

    @property
    def vocab_size(self) -> int:
        return self.blank_idx + 1


class ApexTransducerLoss(torch.nn.Module):
    """
    NVIDIA apex RNNT implementation from the apex module transducer_loss_cuda, or
    the equivalent Triton / TileLang implementation.
    """

    @beartype
    def __init__(
        self,
        backend: Literal["triton", "cuda"] | TileLangKernels,
        blank_idx: int,
        eos_idx: Optional[int],
        star_idx: Optional[int],
        packed_input: bool,
        validate_first_n_remaining: int = 10,
        dp_mu: float = 0.0,
        dp_clamp: Optional[float] = None,
    ):
        super().__init__()
        match backend:
            case "triton":
                self.t_loss = TritonTransducerLoss(
                    packed_input=packed_input, dp_mu=dp_mu, dp_clamp=dp_clamp
                )
            case "cuda":
                assert dp_mu == 0.0 and dp_clamp is None, (
                    "the generalized delay penalty (dp_mu / dp_clamp) is only "
                    "implemented in the triton and tilelang backends"
                )
                self.t_loss = CUDATransducerLoss(packed_input=packed_input)
            case TileLangKernels():
                self.t_loss = TileLangTransducerLoss(
                    kernels=backend,
                    packed_input=packed_input,
                    dp_mu=dp_mu,
                    dp_clamp=dp_clamp,
                )
        self.packed_input = packed_input
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.star_idx = star_idx
        self.validate_first_n_remaining = validate_first_n_remaining

        print_once(
            f"TransducerLoss: backend={backend}, blank_idx={blank_idx}, "
            f"eos_idx={eos_idx}, star_idx={star_idx}"
        )

    @beartype
    def forward(
        self,
        logits: torch.Tensor,
        logit_lens: torch.Tensor,
        y: torch.Tensor,
        y_lens: torch.Tensor,
        batch_offset: Optional[torch.Tensor],
        max_f_len: Optional[int],
        loss_mods: LossModifiers = IDENTITY_LOSS_MODIFIERS,
    ):
        """
        Computes the RNNT loss function

        Args:

            logits: logits tensor of size [B, T, U, K+1] or [batch, max_seq_len,
                max_output_seq, vocab_size+1] if packed_input=False. Otherwise, logits
                is a tensor of size [total_packed, K+1] where
                total_packed = batch_offset[-1].
                Precision is float16, or float32 when `--no_amp` is passed; every
                backend dispatches on it.
            logit_lens: the length of logits tensor.
            y: text transcription tensor.
            y_lens: the length text tensor.
            batch_offset: cumulative sum of T*(U+1).
            max_f_len: max number of features.
            loss_mods: Modifiers for the loss function.
        """

        if y.dtype != torch.int32:
            y = y.int()

        if logit_lens.dtype != torch.int32:
            logit_lens = logit_lens.int()

        if y_lens.dtype != torch.int32:
            y_lens = y_lens.int()

        if self.validate_first_n_remaining > 0:
            # Inputs validation is expensive as it requires sending items from GPU -> CPU
            # and waiting for the result to arrive. Configuring validation to happen
            # only the first validate_first_n_remaining times.
            self._validate_inputs(logits, batch_offset, y)
            self.validate_first_n_remaining -= 1

        loss = self.t_loss(
            logits,
            y,
            logit_lens,
            y_lens,
            self.blank_idx,
            eos_idx=self.eos_idx,
            star_idx=self.star_idx,
            batch_offset=batch_offset,
            max_f_len=max_f_len,
            delay_penalty=loss_mods.dp_lam,
            eos_penalty=loss_mods.eos_lam,
            star_penalty=loss_mods.star_penalty,
        ).mean()

        return loss

    def _validate_inputs(self, logits, batch_offset, y) -> None:
        """
        Validate consistency of inputs.
        """
        if self.packed_input:
            assert len(logits.shape) == 2, (
                "When packed_input=True, logits should be of shape [total_packed, K+1] "
                f"but {logits.shape=}"
            )
            total_packed, _ = logits.shape
            assert total_packed == batch_offset[-1].item(), (
                "Packed input shape and batch_offsets are inconsistent. Should have "
                "total_packed == batch_offset[-1] but "
                f"{total_packed} != {batch_offset[-1].item()}"
            )
        else:
            assert len(logits.shape) == 4, (
                "When packed_input=False, logits should be of shape [B, T, U, K+1] "
                f"but {logits.shape=}"
            )
            assert y.shape[1] == logits.shape[2] - 1, (
                f"When packed_input=False, {y.shape[1]=} should be 1 less than "
                f"{logits.shape[2]=}"
            )


@beartype
class FusedProjAndLoss(torch.nn.Module):
    """An alternative to ``ApexTransducerLoss`` that fuses the joint projection
    into the loss: it takes the already-built, packed, post-ReLU/dropout
    joint activation ``A`` (built by ``rnnt/joint_loss.py``'s
    ``FusedJointLoss`` view) rather than already-projected logits.
    """

    def __init__(
        self,
        kernels: TileLangKernels,
        blank_idx: int,
        eos_idx: Optional[int],
        dp_mu: float = 0.0,
        dp_clamp: Optional[float] = None,
    ):
        super().__init__()
        self.t_loss = TileLangProjAndLoss(
            blank_idx=blank_idx,
            kernels=kernels,
            eos_idx=eos_idx,
            dp_mu=dp_mu,
            dp_clamp=dp_clamp,
        )

        print_once(
            f"TransducerLoss: backend={kernels}, blank_idx={blank_idx}, "
            f"eos_idx={eos_idx}"
        )

    def forward(
        self,
        A: torch.Tensor,
        W: torch.Tensor,
        b: torch.Tensor,
        Label: torch.Tensor,
        f_len: torch.Tensor,
        y_len: torch.Tensor,
        boff: torch.Tensor,
        total: int,
        loss_mods: LossModifiers = IDENTITY_LOSS_MODIFIERS,
    ) -> torch.Tensor:
        return self.t_loss(A, W, b, Label, f_len, y_len, boff, total, loss_mods)


TransducerLossFn = Union[ApexTransducerLoss, FusedProjAndLoss]


@beartype
def _tilelang_kernels(spec: LossSpec) -> TileLangKernels:
    assert spec.device.type == "cuda", "the tilelang joint+loss requires a GPU"
    return select_tilelang_kernels(
        spec.joint_n_hid,
        spec.vocab_size,
        spec.blank_idx,
        spec.device,
        spec.dtype,
    )


@beartype
def prewarm_loss_kernels(spec: LossSpec, dtypes: tuple[torch.dtype, ...]) -> None:
    if spec.backend != "tilelang":
        return
    for dtype in dtypes:
        _tilelang_kernels(replace(spec, dtype=dtype))


@beartype
def build_loss_fn(spec: LossSpec) -> TransducerLossFn:
    if spec.backend != "tilelang":
        return ApexTransducerLoss(
            backend=spec.backend,
            blank_idx=spec.blank_idx,
            eos_idx=spec.eos_idx,
            star_idx=spec.star_idx,
            packed_input=spec.packed_input,
            dp_mu=spec.dp_mu,
            dp_clamp=spec.dp_clamp,
        )

    kernels = _tilelang_kernels(spec)

    if kernels.mode is Mode.fused:
        return FusedProjAndLoss(
            kernels=kernels,
            blank_idx=spec.blank_idx,
            eos_idx=spec.eos_idx,
            dp_mu=spec.dp_mu,
            dp_clamp=spec.dp_clamp,
        )

    return ApexTransducerLoss(
        backend=kernels,
        blank_idx=spec.blank_idx,
        eos_idx=spec.eos_idx,
        star_idx=spec.star_idx,
        packed_input=spec.packed_input,
        dp_mu=spec.dp_mu,
        dp_clamp=spec.dp_clamp,
    )


def get_packing_meta_data(
    feat_lens: torch.Tensor,
    txt_lens: torch.Tensor,
    enc_time_reduction: int,
) -> dict:
    """
    Return packing metadata for TransducerLoss and TransducerJoint
    """
    final_feat_lens = (feat_lens + enc_time_reduction - 1) // enc_time_reduction
    batch_offset = packed_batch_offsets(final_feat_lens, txt_lens + 1)
    max_f_len = max(final_feat_lens).item()

    dict_meta_data = {
        "batch_offset": batch_offset,
        "max_f_len": max_f_len,
        "packed_batch": batch_offset[-1].item(),
    }

    return dict_meta_data
