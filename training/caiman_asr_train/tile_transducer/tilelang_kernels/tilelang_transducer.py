"""
TileLang RNNT: alpha/beta lattice kernel, and the projection-backward kernels.

## Lattice forward/backward (``transducer_loss_fwd``)

This consumes the per-cell emission log-probs produced by ``tilelang_lse``,

    lbx:   (B, T, U)   log P(blank) at (t, u) = x[t, u, blank] - denom[t, u]
    lex:   (B, T, U)   log P(label) at (t, u) = x[t, u, label[u]] - denom[t, u]

The lattice recursion:

    alpha[t, u] = logaddexp(alpha[t-1, u] + lbx[t-1, u],
                            alpha[t, u-1] + lex[t, u-1])
    beta[t, u]  = logaddexp(beta[t+1, u] + lbx[t, u],
                            beta[t, u+1] + lex[t, u])
    loss        = -beta[0, 0]

## Projection-backward kernels (``transducer_loss_bwd`` / ``proj_and_loss_bwd``)

Both read the stored packed logits ``X`` (produced by ``tilelang_lse``'s forward)
and the per-cell coeffs (``compute_coeffs``'s ``c_dense``/``c_blank``/``c_label``) to
form ``dX = f_n(X)`` (cell-wise emission-softmax gradient), launched over the
**packed cell dimension** (one cell = one lattice site ``(b, t, u)``):

  ``transducer_loss_bwd`` (unfused): forms ``dX`` from ``X`` in place;
      Used by ``TileLangTransducerLoss``.

  ``proj_and_loss_bwd`` (fused): forms ``dX``, ``dA = dX @ Wᵀ`` *and* ``db`` in one
      kernel, writing ``dX`` back over ``X`` in place for the torch ``dW``.
      Used by ``TileLangProjAndLoss``.
"""

from collections.abc import Callable

import tilelang
import tilelang.language as T
import torch
from beartype import beartype
from torch import Tensor as Ten

from caiman_asr_train.autotune.dtypes import TLDtype
from caiman_asr_train.tile_transducer.tilelang_kernels.tilelang_math import (
    NEG,
    exp,
    log,
)

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
}


@T.macro
def logaddexp(a, b):
    """log(exp(a) + exp(b)) computed around the running max for stability"""
    m = T.max(a, b)
    return m + log(exp(a - m) + exp(b - m))


@tilelang.jit(pass_configs=PASS_CONFIGS)
@beartype
def transducer_loss_fwd(
    threads: int = 128,
    accum_dtype: TLDtype = torch.float32,
):
    B = T.dynamic("batch")
    max_flen = T.dynamic("max_flen")
    max_glen = T.dynamic("max_glen")
    len_dtype = T.int32
    n_diag = max_flen + max_glen - 1  # number of anti-diagonals

    @T.prim_func
    def main(
        lbx: T.Tensor([B, max_flen, max_glen], accum_dtype),
        lex: T.Tensor([B, max_flen, max_glen], accum_dtype),
        audio_len: T.Tensor([B], len_dtype),
        txt_len: T.Tensor([B], len_dtype),
        alpha: T.Tensor([B, max_flen, max_glen], accum_dtype),
        beta: T.Tensor([B, max_flen, max_glen], accum_dtype),
        loss: T.Tensor([B], accum_dtype),
    ):
        # grid.y = 0 computes alpha, grid.y = 1 computes beta
        with T.Kernel(B, 2, threads=threads) as (bz, alpha_or_beta):
            flen = audio_len[bz]
            glen = txt_len[bz] + 1  # +1 for the SOS slot

            if alpha_or_beta == 0:
                # ---- forward variables ----
                for u in T.Parallel(max_glen):
                    if u == 0:
                        alpha[bz, 0, 0] = 0.0  # empty path has log-prob 0

                for step in T.serial(1, n_diag):
                    for u in T.Parallel(max_glen):
                        t = step - u
                        valid = (u < glen) and (t >= 0) and (t < flen)

                        # Clamp read indices into range; masks below pick NEG
                        # for the transitions that fall off the lattice edge.
                        t_c = T.max(T.min(t, flen - 1), 0)
                        u_c = T.min(u, glen - 1)
                        t_prev = T.max(t_c - 1, 0)
                        u_prev = T.max(u_c - 1, 0)

                        from_blank = T.if_then_else(
                            t > 0,
                            alpha[bz, t_prev, u_c] + lbx[bz, t_prev, u_c],
                            NEG,
                        )
                        from_emit = T.if_then_else(
                            u > 0,
                            alpha[bz, t_c, u_prev] + lex[bz, t_c, u_prev],
                            NEG,
                        )
                        if valid:
                            alpha[bz, t, u] = logaddexp(from_blank, from_emit)
                    T.sync_threads()

            else:
                # ---- backward variables + loss ----
                for u in T.Parallel(max_glen):
                    if u == 0:
                        beta[bz, flen - 1, glen - 1] = lbx[bz, flen - 1, glen - 1]

                # Walk the diagonals in reverse (terminal corner downward).
                for s in T.serial(0, n_diag):
                    step = (n_diag - 1) - s
                    for u in T.Parallel(max_glen):
                        t = step - u
                        valid = (u < glen) and (t >= 0) and (t < flen)
                        is_terminal = (t == flen - 1) and (u == glen - 1)

                        t_c = T.max(T.min(t, flen - 1), 0)
                        u_c = T.min(u, glen - 1)
                        t_next = T.min(t_c + 1, flen - 1)
                        u_next = T.min(u_c + 1, glen - 1)

                        from_blank = T.if_then_else(
                            t < flen - 1,
                            beta[bz, t_next, u_c] + lbx[bz, t_c, u_c],
                            NEG,
                        )
                        from_emit = T.if_then_else(
                            u < glen - 1,
                            beta[bz, t_c, u_next] + lex[bz, t_c, u_c],
                            NEG,
                        )
                        if valid and (not is_terminal):
                            beta[bz, t, u] = logaddexp(from_blank, from_emit)
                    T.sync_threads()

                for u in T.Parallel(max_glen):
                    if u == 0:
                        loss[bz] = -beta[bz, 0, 0]

    return main


@T.macro
def load_bwd_metadata(
    c0,
    n_cells,
    c_dense,
    c_blank,
    c_label,
    denom,
    label_cell,
    cd,
    cb,
    cl,
    dn,
    lab,
    block_C,
):
    """Loads the per-cell loss coefficients and labels into local fragments."""
    if c0 + block_C <= n_cells:
        T.copy(c_dense[c0 : c0 + block_C], cd)
        T.copy(c_blank[c0 : c0 + block_C], cb)
        T.copy(c_label[c0 : c0 + block_C], cl)
        T.copy(denom[c0 : c0 + block_C], dn)
        T.copy(label_cell[c0 : c0 + block_C], lab)
    else:
        for c in T.Parallel(block_C):
            cc = T.min(c0 + c, n_cells - 1)
            cd[c] = c_dense[cc]
            cb[c] = c_blank[cc]
            cl[c] = c_label[cc]
            dn[c] = denom[cc]
            lab[c] = label_cell[cc]


@T.macro
def compute_dx(v0, blank_idx, x_in, x_out, dn, cd, cb, cl, lab, block_C, block_V):
    """Computes the emission-softmax gradient (dX) for a given tile."""
    for c, j in T.Parallel(block_C, block_V):
        gv = v0 + j
        sv = exp(x_in[c, j] - dn[c])
        x_out[c, j] = (
            cd[c] * sv
            - T.if_then_else(gv == blank_idx, cb[c], 0.0)
            - T.if_then_else(gv == lab[c], cl[c], 0.0)
        )


i32 = T.int32

FUSED_PASS_CONFIGS = PASS_CONFIGS | {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


# ===========================================================================
# transducer_loss_bwd (unfused, ex-naiive_kernel1): cell-wise dX over packed
# cells. Used by TileLangTransducerLoss,.
# ===========================================================================
@tilelang.jit(pass_configs=PASS_CONFIGS)
@beartype
def transducer_loss_bwd(
    V: int,
    blank_idx: int,
    block_C: int = 128,
    block_V: int = 128,
    threads: int = 128,
    dtype: TLDtype = torch.float16,
    accum_dtype: TLDtype = torch.float32,
):
    """
    Read the pre-stored X, compute dX (in place).

    Pseudocode:
    ```
    Parallel (n_cells/b_C, V/b_V):
        load per-cell scalars: c_dense, c_blank, c_label, denom, label
        load X_tile <- X
        dX_tile = dLdx(X_tile)
        store dX_tile -> X
    ```
    """
    n_cells = T.dynamic("n_cells")
    n_v = (V + block_V - 1) // block_V
    v_tail = V % block_V != 0

    @T.prim_func
    def main(
        c_dense: T.Tensor([n_cells], accum_dtype),
        c_blank: T.Tensor([n_cells], accum_dtype),
        c_label: T.Tensor([n_cells], accum_dtype),
        denom: T.Tensor([n_cells], accum_dtype),
        label_cell: T.Tensor([n_cells], i32),
        X: T.Tensor([n_cells, V], dtype),  # in: logits; out: dX (in place)
    ):
        with T.Kernel(T.ceildiv(n_cells, block_C) * n_v, threads=threads) as (bid,):
            bc, bv = bid // n_v, bid % n_v
            x_shared = T.alloc_shared([block_C, block_V], dtype)
            x = T.alloc_fragment([block_C, block_V], accum_dtype)
            cd = T.alloc_fragment([block_C], accum_dtype)
            cb = T.alloc_fragment([block_C], accum_dtype)
            cl = T.alloc_fragment([block_C], accum_dtype)
            dn = T.alloc_fragment([block_C], accum_dtype)
            lab = T.alloc_fragment([block_C], i32)

            c0, v0 = bc * block_C, bv * block_V
            load_bwd_metadata(
                c0,
                n_cells,
                c_dense,
                c_blank,
                c_label,
                denom,
                label_cell,
                cd,
                cb,
                cl,
                dn,
                lab,
                block_C,
            )

            if v_tail:
                for c, j in T.Parallel(block_C, block_V):
                    x_shared[c, j] = X[T.min(c0 + c, n_cells - 1), T.min(v0 + j, V - 1)]
            else:
                if c0 + block_C <= n_cells:
                    T.copy(X[c0 : c0 + block_C, v0 : v0 + block_V], x_shared)
                else:
                    for c, j in T.Parallel(block_C, block_V):
                        x_shared[c, j] = X[T.min(c0 + c, n_cells - 1), v0 + j]
            T.copy(x_shared, x)

            compute_dx(v0, blank_idx, x, x, dn, cd, cb, cl, lab, block_C, block_V)

            T.copy(x, x_shared)
            if v_tail:
                for c, j in T.Parallel(block_C, block_V):
                    if (c0 + c < n_cells) and (v0 + j < V):
                        X[c0 + c, v0 + j] = x_shared[c, j]
            else:
                if c0 + block_C <= n_cells:
                    T.copy(x_shared, X[c0 : c0 + block_C, v0 : v0 + block_V])
                else:
                    for c, j in T.Parallel(block_C, block_V):
                        if c0 + c < n_cells:
                            X[c0 + c, v0 + j] = x_shared[c, j]

    return main


@beartype
def transducer_loss_bwd_run(kernel: Callable):
    def run(
        c_dense: Ten, c_blank: Ten, c_label: Ten, denom: Ten, label_cell: Ten, X: Ten
    ):
        kernel(c_dense, c_blank, c_label, denom, label_cell, X)
        return X

    return run


# ===========================================================================
# proj_and_loss_bwd: one kernel forms dX, dA = dX @ Wᵀ and db over packed cells.
# Used by TileLangProjAndLoss.
# ===========================================================================
@tilelang.jit(pass_configs=FUSED_PASS_CONFIGS)
@beartype
def proj_and_loss_bwd(
    H: int,
    V: int,
    blank_idx: int,
    block_C: int = 128,
    block_V: int = 64,
    block_H: int = 64,
    threads: int = 128,
    num_stages: int = 2,
    dtype: TLDtype = torch.float16,
    accum_dtype: TLDtype = torch.float32,
):
    """
    Read the pre-stored X, compute dA = dLdx(X) @ Wᵀ and db, writing dX over X on the
    last H tile. dA is returned UNMASKED (Apex re-applies the ReLU + dropout masks).

    Pseudocode:
    ```
    Parallel (n_cells/b_C):
        load per-cell scalars: c_dense, c_blank, c_label, denom, label
        Serial (H/b_H):
            da_acc = 0
            Pipelined (V/b_V):
                load X_tile <- X , W_tile <- W
                dX_tile = dLdx(X_tile)
                if last h: store dX_tile -> X            (in place)
                if last h: db[v_tile] += colsum(dX_tile) (atomic)
                da_acc += dX_tile @ W_tile.T
            store da_acc -> dA[:, h_tile]                (unmasked)
    ```
    """
    n_cells = T.dynamic("n_cells")
    n_h = (H + block_H - 1) // block_H
    v_tail = V % block_V != 0
    h_tail = H % block_H != 0

    @T.prim_func
    def main(
        c_dense: T.Tensor([n_cells], accum_dtype),
        c_blank: T.Tensor([n_cells], accum_dtype),
        c_label: T.Tensor([n_cells], accum_dtype),
        denom: T.Tensor([n_cells], accum_dtype),
        label_cell: T.Tensor([n_cells], i32),
        X: T.Tensor([n_cells, V], dtype),  # in: logits; out: dX (in place)
        W: T.Tensor([H, V], dtype),
        dA: T.Tensor([n_cells, H], dtype),  # out (unmasked dX@Wᵀ)
        db: T.Tensor([V], accum_dtype),  # out, zero on entry
    ):
        with T.Kernel(T.ceildiv(n_cells, block_C), threads=threads) as (bc,):
            x_shared = T.alloc_shared([block_C, block_V], dtype)
            w_shared = T.alloc_shared([block_H, block_V], dtype)
            x = T.alloc_fragment([block_C, block_V], accum_dtype)
            dx_cast = T.alloc_fragment([block_C, block_V], dtype)
            da_acc = T.alloc_fragment([block_C, block_H], accum_dtype)
            da_cast = T.alloc_fragment([block_C, block_H], dtype)
            db_blk = T.alloc_fragment([block_V], accum_dtype)
            cd = T.alloc_fragment([block_C], accum_dtype)
            cb = T.alloc_fragment([block_C], accum_dtype)
            cl = T.alloc_fragment([block_C], accum_dtype)
            dn = T.alloc_fragment([block_C], accum_dtype)
            lab = T.alloc_fragment([block_C], i32)

            c0 = bc * block_C
            load_bwd_metadata(
                c0,
                n_cells,
                c_dense,
                c_blank,
                c_label,
                denom,
                label_cell,
                cd,
                cb,
                cl,
                dn,
                lab,
                block_C,
            )

            for hb in T.serial(n_h):
                h0 = hb * block_H
                T.clear(da_acc)
                for vb in T.Pipelined(T.ceildiv(V, block_V), num_stages=num_stages):
                    v0 = vb * block_V
                    if v_tail:
                        for c, j in T.Parallel(block_C, block_V):
                            x_shared[c, j] = X[
                                T.min(c0 + c, n_cells - 1), T.min(v0 + j, V - 1)
                            ]
                    else:
                        if c0 + block_C <= n_cells:
                            T.copy(X[c0 : c0 + block_C, v0 : v0 + block_V], x_shared)
                        else:
                            for c, j in T.Parallel(block_C, block_V):
                                x_shared[c, j] = X[T.min(c0 + c, n_cells - 1), v0 + j]
                    if v_tail or h_tail:
                        for k, j in T.Parallel(block_H, block_V):
                            w_shared[k, j] = W[
                                T.min(h0 + k, H - 1), T.min(v0 + j, V - 1)
                            ]
                    else:
                        T.copy(W[h0 : h0 + block_H, v0 : v0 + block_V], w_shared)

                    compute_dx(
                        v0,
                        blank_idx,
                        x_shared,
                        x,
                        dn,
                        cd,
                        cb,
                        cl,
                        lab,
                        block_C,
                        block_V,
                    )

                    if v_tail:
                        for c, j in T.Parallel(block_C, block_V):
                            if v0 + j >= V:
                                x[c, j] = 0.0

                    if c0 + block_C > n_cells:
                        for c, j in T.Parallel(block_C, block_V):
                            if c0 + c >= n_cells:
                                x[c, j] = 0.0

                    T.copy(x, dx_cast)

                    T.gemm(dx_cast, w_shared, da_acc, transpose_B=True)

                    if hb == n_h - 1:
                        T.reduce_sum(x, db_blk, dim=0)
                        if v_tail:
                            for j in T.Parallel(block_V):
                                if v0 + j < V:
                                    T.atomic_add(db[v0 + j], db_blk[j])
                            for c, j in T.Parallel(block_C, block_V):
                                if (c0 + c < n_cells) and (v0 + j < V):
                                    X[c0 + c, v0 + j] = dx_cast[c, j]
                        else:
                            T.atomic_add(db[v0 : v0 + block_V], db_blk)
                            if c0 + block_C <= n_cells:
                                T.copy(
                                    dx_cast,
                                    X[c0 : c0 + block_C, v0 : v0 + block_V],
                                )
                            else:
                                for c, j in T.Parallel(block_C, block_V):
                                    if c0 + c < n_cells:
                                        X[c0 + c, v0 + j] = dx_cast[c, j]

                T.copy(da_acc, da_cast)
                if h_tail:
                    for c, h in T.Parallel(block_C, block_H):
                        if (c0 + c < n_cells) and (h0 + h < H):
                            dA[c0 + c, h0 + h] = da_cast[c, h]
                else:
                    if c0 + block_C <= n_cells:
                        T.copy(da_cast, dA[c0 : c0 + block_C, h0 : h0 + block_H])
                    else:
                        for c, h in T.Parallel(block_C, block_H):
                            if c0 + c < n_cells:
                                dA[c0 + c, h0 + h] = da_cast[c, h]

    return main


@beartype
def proj_and_loss_bwd_run(
    H: int,
    V: int,
    kernel: Callable,
    dtype: torch.dtype = torch.float16,
    accum_dtype: torch.dtype = torch.float32,
):
    def run(
        A_c: Ten,
        W: Ten,
        c_dense: Ten,
        c_blank: Ten,
        c_label: Ten,
        denom: Ten,
        label_cell: Ten,
        X: Ten,
        total: int,
    ):
        dev, n_cells = X.device, X.shape[0]

        dA = torch.empty(n_cells, H, device=dev, dtype=dtype)
        db = torch.zeros(V, device=dev, dtype=accum_dtype)
        kernel(c_dense, c_blank, c_label, denom, label_cell, X, W, dA, db)
        dW = torch.matmul(A_c[:total].t(), X[:total])  # (H, V)
        return dA, dW, db

    return run
