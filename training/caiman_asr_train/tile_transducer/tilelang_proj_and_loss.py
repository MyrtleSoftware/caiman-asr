"""Tilelang implementation of the RNN-T FusedProjAndLoss."""

import torch
from beartype import beartype
from beartype.typing import Optional
from torch import Tensor as Ten

from caiman_asr_train.rnnt.loss_modifiers import LossModifiers
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


class TileLangProjAndLossFunc(torch.autograd.Function):
    """Projection (``X = A@W+b``) + fused logsumexp + alpha/beta lattice (forward,
    stores ``X``); store-X 2-GEMM projection backward returning ``(dA, dW, db)``."""

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.amp.custom_fwd(device_type="cuda")
    @beartype
    def forward(
        ctx,
        A: Ten,
        W: Ten,
        b: Ten,
        Label: Ten,
        f_len: Ten,
        y_len: Ten,
        boff: Ten,
        total: int,
        blank_idx: int,
        fwd_run,
        bwd_run,
        eos_idx: Optional[int],
        dp_lam: float,
        eos_lam: float,
        dp_mu: float,
        dp_gamma: float,
        accum_dtype: torch.dtype,
    ):
        device = A.device
        batch = f_len.shape[0]
        H, V = A.shape[1], W.shape[1]
        f_len, y_len, Label = f_len.int(), y_len.int(), Label.int()

        max_flen = int(f_len.max().item())
        max_glen = Label.shape[1] + 1  # >= max(y_len)+1; Label width = max_glen-1

        Xoff = torch.zeros(batch, device=device, dtype=torch.int64)
        Xoff[1:] = boff[:-1]

        # The forward's cell-launch kernels (proj_lse) carry no (b,t,u) of their
        # own, so the packed-cell <-> rectangular-cell metadata is built once, up
        # front, and reused by both the forward's lbx/lex scatter and the
        # backward's per-cell coeff gather.
        idx_flat = build_cell_index(f_len, y_len, Xoff, max_flen, max_glen, total)
        label_cell = build_label_cell(Label, max_flen, max_glen, idx_flat)

        denom_p = torch.zeros(total, device=device, dtype=accum_dtype)
        lbx, lex, alpha, beta = (
            torch.zeros(batch, max_flen, max_glen, device=device, dtype=accum_dtype)
            for _ in range(4)
        )
        loss = torch.empty(batch, device=device, dtype=accum_dtype)
        X = torch.empty(total, V, device=device, dtype=A.dtype)

        fwd_run(A, W, b, label_cell, idx_flat, denom_p, lbx, lex, X, total)
        apply_emit_penalty(lex, f_len, Label, dp_lam, eos_lam, eos_idx, dp_mu, dp_gamma)
        transducer_loss_fwd(accum_dtype=accum_dtype)(
            lbx, lex, f_len, y_len, alpha, beta, loss
        )

        ctx.save_for_backward(A, W, alpha, beta, lbx, lex, f_len, y_len)
        ctx.X, ctx.denom_p, ctx.label_cell, ctx.idx_flat = (
            X,
            denom_p,
            label_cell,
            idx_flat,
        )
        ctx.blank_idx, ctx.bwd_run = blank_idx, bwd_run
        ctx.H, ctx.V, ctx.total = H, V, total
        return loss

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.amp.custom_bwd(device_type="cuda")
    @beartype
    def backward(ctx, loss_grad: Ten):
        A, W, alpha, beta, lbx, lex, f_len, y_len = ctx.saved_tensors

        c_dense, c_blank, c_label = compute_coeffs(
            alpha, beta, lbx, lex, f_len, y_len, loss_grad
        )
        cd = pack_scalar(c_dense, ctx.idx_flat)
        cb = pack_scalar(c_blank, ctx.idx_flat)
        cl = pack_scalar(c_label, ctx.idx_flat)

        dA, dW, db = ctx.bwd_run(
            A,
            W,
            cd,
            cb,
            cl,
            ctx.denom_p,
            ctx.label_cell,
            ctx.X,
            ctx.total,
        )
        return dA.to(A.dtype), dW.to(W.dtype), db.to(W.dtype), *([None] * 14)


@beartype
class TileLangProjAndLoss(torch.nn.Module):
    """The TileLang backend for the fused joint-projection+loss. Returns a
    differentiable per-sequence loss ``(B,)``;"""

    def __init__(
        self,
        blank_idx: int,
        kernels: TileLangKernels,
        eos_idx: Optional[int] = None,
        dp_mu: float = 0.0,
        dp_clamp: Optional[float] = None,
    ):
        super().__init__()
        self.blank_idx = blank_idx
        self.kernels = kernels
        self.eos_idx = eos_idx
        self.dp_mu = dp_mu
        self.dp_gamma = dp_clamp if dp_clamp is not None else NO_CLAMP

    @beartype
    def forward(
        self,
        A: Ten,
        W: Ten,
        b: Ten,
        Label: Ten,
        f_len: Ten,
        y_len: Ten,
        boff: Ten,
        total: int,
        loss_mods: LossModifiers,
    ):
        kernels = self.kernels
        assert A.dtype is kernels.dtype, (
            f"joint activation arrived as {A.dtype} but these kernels were tuned for "
            f"{kernels.dtype}; the loss is built per precision at setup"
        )
        A = A.contiguous()
        # tilelang enforces a packed (row-major) ABI, so W must be materialised:
        # the training path passes ``joint_net[-1].weight.t()``, a transposed view.
        W = W.to(A.dtype).contiguous()
        b = b.to(A.dtype)

        return TileLangProjAndLossFunc.apply(
            A,
            W,
            b,
            Label,
            f_len,
            y_len,
            boff,
            total,
            self.blank_idx,
            kernels.fwd_run,
            kernels.bwd_run,
            self.eos_idx,
            loss_mods.dp_lam,
            loss_mods.eos_lam,
            self.dp_mu,
            self.dp_gamma,
            kernels.accum_dtype,
        )
