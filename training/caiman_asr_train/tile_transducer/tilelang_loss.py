"""Tiellang Implementation of the (unfused) RNN-T Transducer Loss"""

import torch
from beartype import beartype
from beartype.typing import List, Optional
from torch import Tensor as Ten

from caiman_asr_train.tile_transducer.autotune.select import TileLangKernels
from caiman_asr_train.tile_transducer.tilelang_kernels.tilelang_transducer import (
    transducer_loss_fwd,
)
from caiman_asr_train.tile_transducer.utils import (
    NO_CLAMP,
    apply_emit_penalty,
    build_cell_index,
    build_label_cell,
    compute_coeffs,
    pack_scalar,
)


class TileLangTransducerLossFunc(torch.autograd.Function):
    """Logsumexp + alpha/beta lattice (forward); ``dX``-only projection backward,
    formed in place over the caller's own logits, so the returned gradient
    aliases ``x``."""

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.amp.custom_fwd(device_type="cuda")
    @beartype
    def forward(
        ctx,
        x: Ten,
        label: Ten,
        f_len: Ten,
        y_len: Ten,
        blank_idx: int,
        batch_offset: Ten,
        max_f_len: int,
        packed_input: bool,
        fwd_run,
        bwd_run,
        eos_idx: Optional[int],
        dp_lam: float,
        eos_lam: float,
        dp_mu: float,
        dp_gamma: float,
        accum_dtype: torch.dtype,
    ):
        device = x.device
        batch = f_len.shape[0]
        f_len, y_len, label = f_len.int(), y_len.int(), label.int()
        max_glen = label.shape[1] + 1  # >= max(y_len)+1; Label width = max_glen-1

        if packed_input:
            V = x.shape[-1]
            boff = batch_offset.to(torch.int64)
            total = int(boff[-1].item())
            max_flen = max_f_len
            x_flat = x
        else:
            B, T, U, V = x.shape
            max_flen = T
            # No ragged gaps: a uniform-stride "pack" of the whole dense box.
            boff = torch.arange(1, B + 1, device=device, dtype=torch.int64) * (T * U)
            total = B * T * U
            x_flat = x.reshape(total, V)

        Xoff = torch.zeros(batch, device=device, dtype=torch.int64)
        Xoff[1:] = boff[:-1]

        # The forward's cell-launch kernel (logsumexp) carries no (b,t,u) of its
        # own, so the packed-cell <-> rectangular-cell metadata is built once, up
        # front, and reused by both the forward's lbx/lex scatter and the
        # backward's per-cell coeff gather.
        idx_flat = build_cell_index(f_len, y_len, Xoff, max_flen, max_glen, total)
        label_cell = build_label_cell(label, max_flen, max_glen, idx_flat)

        # A no-op for the contiguous common case, so the returned grad aliases x.
        X = x_flat.contiguous()

        denom_p = torch.zeros(total, device=device, dtype=accum_dtype)
        lbx, lex, alpha, beta = (
            torch.zeros(batch, max_flen, max_glen, device=device, dtype=accum_dtype)
            for _ in range(4)
        )
        loss = torch.empty(batch, device=device, dtype=accum_dtype)

        fwd_run(X, label_cell, idx_flat, denom_p, lbx, lex)
        apply_emit_penalty(lex, f_len, label, dp_lam, eos_lam, eos_idx, dp_mu, dp_gamma)
        transducer_loss_fwd(accum_dtype=accum_dtype)(
            lbx, lex, f_len, y_len, alpha, beta, loss
        )

        ctx.save_for_backward(alpha, beta, lbx, lex, f_len, y_len)
        ctx.X, ctx.denom_p, ctx.label_cell, ctx.idx_flat = (
            X,
            denom_p,
            label_cell,
            idx_flat,
        )
        ctx.bwd_run = bwd_run
        ctx.packed_input, ctx.x_shape = packed_input, x.shape
        return loss

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.amp.custom_bwd(device_type="cuda")
    @beartype
    def backward(ctx, loss_grad: Ten):
        alpha, beta, lbx, lex, f_len, y_len = ctx.saved_tensors

        c_dense, c_blank, c_label = compute_coeffs(
            alpha, beta, lbx, lex, f_len, y_len, loss_grad
        )
        cd = pack_scalar(c_dense, ctx.idx_flat)
        cb = pack_scalar(c_blank, ctx.idx_flat)
        cl = pack_scalar(c_label, ctx.idx_flat)

        X, ctx.X = ctx.X, None
        dx = ctx.bwd_run(cd, cb, cl, ctx.denom_p, ctx.label_cell, X)  # in place

        if not ctx.packed_input:
            dx = dx.reshape(ctx.x_shape)
        return dx, *([None] * 15)


@beartype
class TileLangTransducerLoss(torch.nn.Module):
    """A TileLang implementation of the RNN-T loss, mirroring
    ``TritonTransducerLoss``/``CUDATransducerLoss``."""

    def __init__(
        self,
        kernels: TileLangKernels,
        packed_input: bool = False,
        dp_mu: float = 0.0,
        dp_clamp: Optional[float] = None,
    ):
        super().__init__()
        self.packed_input = packed_input
        self.kernels = kernels
        self.dp_mu = dp_mu
        self.dp_gamma = dp_clamp if dp_clamp is not None else NO_CLAMP

    @beartype
    def forward(
        self,
        x: Ten,
        label: Ten,
        f_len: Ten,
        y_len: Ten,
        blank_idx: int,
        eos_idx: Optional[int] = None,
        star_idx: Optional[int] = None,
        batch_offset: Optional[Ten] = None,
        max_f_len: Optional[int] = None,
        debug_list: Optional[List[Ten]] = None,
        delay_penalty: float = 0.0,
        eos_penalty: float = 0.0,
        star_penalty: float = 1.0,
    ) -> Ten:
        assert len(x.shape) == 4 or len(x.shape) == 2, "Shape (B, T, U, H) or (*, H)"
        assert star_idx is None, "star penalty is not supported"
        assert f_len.min() >= 1, "f_len must be non-negative"
        assert y_len.min() >= 0, "y_len must be non-negative"
        assert y_len.max() <= label.size(1), "y_len must be less than label length"

        if self.packed_input:
            if batch_offset is None or max_f_len is None:
                raise Exception(
                    "Please specify batch_offset and max_f_len when packing is enabled"
                )
            assert max_f_len == f_len.max()

        kernels = self.kernels
        assert x.dtype is kernels.dtype, (
            f"logits arrived as {x.dtype} but these kernels were tuned for "
            f"{kernels.dtype}; the loss is built per precision at setup"
        )

        return TileLangTransducerLossFunc.apply(
            x,
            label,
            f_len,
            y_len,
            blank_idx,
            batch_offset,
            max_f_len,
            self.packed_input,
            kernels.fwd_run,
            kernels.bwd_run,
            eos_idx,
            delay_penalty,
            eos_penalty,
            self.dp_mu,
            self.dp_gamma,
            kernels.accum_dtype,
        )
