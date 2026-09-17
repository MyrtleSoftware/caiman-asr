"""
TileLang RNN-T forward kernels: projection + logsumexp of the joint network output.

Two kernels, used by two different loss modules:

  ``proj_lse`` (used by ``TileLangProjAndLoss``, the fused mode): a single tilelang
      kernel forms the projection ``X = A@W+b`` and does the online logsumexp,
      storing the packed logits ``X`` for the backward.

  ``logsumexp`` (used by ``TileLangTransducerLoss``, the unfused mode): reads an
      already-materialised ``X`` and does only the online logsumexp
      + blank/label numerators.

Inputs:
    A: (n_cells, H)         packed post-ReLU/dropout joint activation (Apex output)
    W: (H, V)               joint projection weight   (V = vocab)
    b: (V,)                 joint projection bias
    LabelCell: (n_cells,)   int32 label token of each packed cell
    CellIdx: (n_cells,)     int64 flat (b,t,u) index of each packed cell


Results (per packed cell r; lbx/lex are scattered into the rectangular (B, T, U)
lattice box via CellIdx):
    Denom[r] = logsumexp(X[r], dim=-1)          the log-normaliser, PACKED (n_cells,)
    lbx[cell(r)] = X[r, blank] - Denom[r]        log P(blank)
    lex[cell(r)] = X[r, label] - Denom[r]        log P(label), UNPENALISED..
    X[r]  = A[r] @ W + b                        stored packed for the backward
"""

from collections.abc import Callable

import tilelang
import tilelang.language as T
import torch
from beartype import beartype

from caiman_asr_train.autotune.dtypes import TLDtype
from caiman_asr_train.tile_transducer.tilelang_kernels.tilelang_math import (
    NEG,
    exp,
    log,
)

_SCATTER_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@T.macro
def prologue_init(
    c0,
    n_cells,
    LabelCell,
    CellIdx,
    lab,
    ridx,
    lbx_frag,
    lex_frag,
    logsum,
    scores_max,
    accum_dtype,
    block_C,
):
    """Loads cell metadata (labels, scatter indices) and initializes accumulators."""
    if c0 + block_C <= n_cells:
        T.copy(LabelCell[c0 : c0 + block_C], lab)
        T.copy(CellIdx[c0 : c0 + block_C], ridx)
    else:
        for i in T.Parallel(block_C):
            ci = T.min(c0 + i, n_cells - 1)
            lab[i] = LabelCell[ci]
            ridx[i] = CellIdx[ci]

    T.clear(lbx_frag)
    T.clear(lex_frag)
    T.clear(logsum)
    T.fill(scores_max, -T.infinity(accum_dtype))


@T.macro
def accum_lbx_lex(
    v0,
    blank_idx,
    acc_s,
    lab,
    masked_logits,
    block_logit,
    lbx_frag,
    lex_frag,
    block_C,
    block_V,
):
    """Extracts and accumulates the raw logits for the blank and label."""
    # log P(blank) numerator
    for i, j in T.Parallel(block_C, block_V):
        masked_logits[i, j] = T.if_then_else(v0 + j == blank_idx, acc_s[i, j], 0.0)
    T.reduce_sum(masked_logits, block_logit, dim=1)
    for i in T.Parallel(block_C):
        lbx_frag[i] = lbx_frag[i] + block_logit[i]

    # log P(label) numerator
    for i, j in T.Parallel(block_C, block_V):
        masked_logits[i, j] = T.if_then_else(v0 + j == lab[i], acc_s[i, j], 0.0)
    T.reduce_sum(masked_logits, block_logit, dim=1)
    for i in T.Parallel(block_C):
        lex_frag[i] = lex_frag[i] + block_logit[i]


@T.macro
def update_online_lse(
    acc_s,
    scores_max,
    scores_max_prev,
    scores_scale,
    scores_sum,
    logsum,
    block_C,
    block_V,
):
    """Performs a block-wise update for the online LogSumExp running max and sum."""
    # Online logsumexp over the vocab blocks.
    T.copy(scores_max, scores_max_prev)
    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
    for i in T.Parallel(block_C):
        scores_scale[i] = exp(scores_max_prev[i] - scores_max[i])
    for i, j in T.Parallel(block_C, block_V):
        acc_s[i, j] = exp(acc_s[i, j] - scores_max[i])
    T.reduce_sum(acc_s, scores_sum, dim=1)
    for i in T.Parallel(block_C):
        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]


@T.macro
def epilogue_norm_scatter(
    lse,
    scores_max,
    logsum,
    lbx_frag,
    lex_frag,
    Denom,
    Lbx,
    Lex,
    ridx,
    c0,
    n_cells,
    block_C,
):
    """Finalizes LogSumExp, normalizes logits to log-probs,
    and scatters them into the lattice."""
    # logsumexp = running_max + ln(running_sum_of_exp)
    for i in T.Parallel(block_C):
        lse[i] = scores_max[i] + log(logsum[i])

    # Normalise the numerators by the denom -> log P, then scatter the
    # in-range cells into the rectangular (B, T, U) lattice box.
    for i in T.Parallel(block_C):
        lbx_frag[i] = lbx_frag[i] - lse[i]
        lex_frag[i] = lex_frag[i] - lse[i]

    for i in T.Parallel(block_C):
        if c0 + i < n_cells:
            Denom[c0 + i] = lse[i]
            Lbx[ridx[i]] = lbx_frag[i]
            Lex[ridx[i]] = lex_frag[i]


@tilelang.jit(pass_configs=_SCATTER_PASS_CONFIGS)
@beartype
def proj_lse(
    H: int,
    V: int,
    blank_idx: int,
    block_C: int = 128,
    block_V: int = 128,
    block_H: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    dtype: TLDtype = torch.float16,
    accum_dtype: TLDtype = torch.float32,
):
    """Fused projection + online logsumexp; see the module docstring for detail.

    Torch Equivalent: ``X = A @ W + b; Denom = X.logsumexp(-1)``
    """
    n_cells = T.dynamic("n_cells")  # packed cells (A / X / per-cell metadata)
    n_rect = T.dynamic("n_rect")  # B * max_flen * max_glen (rect lbx/lex, flat)
    label_dtype = T.int32
    idx_dtype = T.int64

    v_tail = V % block_V != 0
    h_tail = H % block_H != 0

    @T.prim_func
    def main(
        A: T.Tensor([n_cells, H], dtype),  # packed post-ReLU/dropout joint (Apex)
        W: T.Tensor([H, V], dtype),
        Bias: T.Tensor([V], dtype),
        LabelCell: T.Tensor([n_cells], label_dtype),  # per-cell label token
        CellIdx: T.Tensor([n_cells], idx_dtype),  # per-cell flat (b,t,u)
        Denom: T.Tensor([n_cells], accum_dtype),  # packed logsumexp, out
        Lbx: T.Tensor([n_rect], accum_dtype),  # rect (B,T,U) flat, scattered out
        Lex: T.Tensor([n_rect], accum_dtype),  # rect (B,T,U) flat, scattered out
        X: T.Tensor([n_cells, V], dtype),  # packed logits store (for the backward)
    ):
        with T.Kernel(T.ceildiv(n_cells, block_C), threads=threads) as (bc,):
            a_shared = T.alloc_shared([block_C, block_H], dtype)
            w_shared = T.alloc_shared([block_H, block_V], dtype)
            b_shared = T.alloc_shared([block_V], dtype)

            acc_s = T.alloc_fragment([block_C, block_V], accum_dtype)  # joint block
            scores_max = T.alloc_fragment([block_C], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_C], accum_dtype)
            scores_scale = T.alloc_fragment([block_C], accum_dtype)
            scores_sum = T.alloc_fragment([block_C], accum_dtype)
            logsum = T.alloc_fragment([block_C], accum_dtype)
            lse = T.alloc_fragment([block_C], accum_dtype)
            masked_logits = T.alloc_fragment([block_C, block_V], accum_dtype)
            block_logit = T.alloc_fragment([block_C], accum_dtype)
            lbx_frag = T.alloc_fragment([block_C], accum_dtype)
            lex_frag = T.alloc_fragment([block_C], accum_dtype)
            # accum acc_s -> storage-dtype staging before the X store.
            x_store = T.alloc_fragment([block_C, block_V], dtype)
            lab = T.alloc_fragment([block_C], label_dtype)  # per-cell label token
            ridx = T.alloc_fragment([block_C], idx_dtype)  # per-cell scatter target

            c0 = bc * block_C
            prologue_init(
                c0=c0,
                n_cells=n_cells,
                LabelCell=LabelCell,
                CellIdx=CellIdx,
                lab=lab,
                ridx=ridx,
                lbx_frag=lbx_frag,
                lex_frag=lex_frag,
                logsum=logsum,
                scores_max=scores_max,
                accum_dtype=accum_dtype,
                block_C=block_C,
            )

            for v in T.Serial(T.ceildiv(V, block_V)):
                if v_tail:
                    for j in T.Parallel(block_V):
                        b_shared[j] = Bias[T.min(v * block_V + j, V - 1)]
                else:
                    T.copy(Bias[v * block_V : (v + 1) * block_V], b_shared)
                # Prefill acc_s with the bias.
                for i, j in T.Parallel(block_C, block_V):
                    acc_s[i, j] = b_shared[j]

                # Accumulate A @ W over H-blocks.
                for h in T.Pipelined(T.ceildiv(H, block_H), num_stages=num_stages):
                    if h_tail:
                        for i, k in T.Parallel(block_C, block_H):
                            a_shared[i, k] = T.if_then_else(
                                h * block_H + k < H,
                                A[
                                    T.min(c0 + i, n_cells - 1),
                                    T.min(h * block_H + k, H - 1),
                                ],
                                0,
                            )
                    else:
                        if c0 + block_C <= n_cells:
                            T.copy(
                                A[
                                    c0 : c0 + block_C,
                                    h * block_H : (h + 1) * block_H,
                                ],
                                a_shared,
                            )
                        else:
                            for i, k in T.Parallel(block_C, block_H):
                                a_shared[i, k] = A[
                                    T.min(c0 + i, n_cells - 1), h * block_H + k
                                ]
                    if v_tail or h_tail:
                        for k, j in T.Parallel(block_H, block_V):
                            w_shared[k, j] = W[
                                T.min(h * block_H + k, H - 1),
                                T.min(v * block_V + j, V - 1),
                            ]
                    else:
                        T.copy(
                            W[
                                h * block_H : (h + 1) * block_H,
                                v * block_V : (v + 1) * block_V,
                            ],
                            w_shared,
                        )
                    T.gemm(
                        a_shared,
                        w_shared,
                        acc_s,
                        clear_accum=False,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                if v_tail:
                    for i, j in T.Parallel(block_C, block_V):
                        if v * block_V + j >= V:
                            acc_s[i, j] = NEG

                accum_lbx_lex(
                    v0=v * block_V,
                    blank_idx=blank_idx,
                    acc_s=acc_s,
                    lab=lab,
                    masked_logits=masked_logits,
                    block_logit=block_logit,
                    lbx_frag=lbx_frag,
                    lex_frag=lex_frag,
                    block_C=block_C,
                    block_V=block_V,
                )

                # Store the raw joint block (accum -> storage dtype) for the backward.
                T.copy(acc_s, x_store)
                if v_tail:
                    for i, j in T.Parallel(block_C, block_V):
                        if (c0 + i < n_cells) and (v * block_V + j < V):
                            X[c0 + i, v * block_V + j] = x_store[i, j]
                else:
                    if c0 + block_C <= n_cells:
                        T.copy(
                            x_store,
                            X[c0 : c0 + block_C, v * block_V : (v + 1) * block_V],
                        )
                    else:
                        for i, j in T.Parallel(block_C, block_V):
                            if c0 + i < n_cells:
                                X[c0 + i, v * block_V + j] = x_store[i, j]

                update_online_lse(
                    acc_s=acc_s,
                    scores_max=scores_max,
                    scores_max_prev=scores_max_prev,
                    scores_scale=scores_scale,
                    scores_sum=scores_sum,
                    logsum=logsum,
                    block_C=block_C,
                    block_V=block_V,
                )

            epilogue_norm_scatter(
                lse=lse,
                scores_max=scores_max,
                logsum=logsum,
                lbx_frag=lbx_frag,
                lex_frag=lex_frag,
                Denom=Denom,
                Lbx=Lbx,
                Lex=Lex,
                ridx=ridx,
                c0=c0,
                n_cells=n_cells,
                block_C=block_C,
            )

    return main


@tilelang.jit(pass_configs=_SCATTER_PASS_CONFIGS)
@beartype
def logsumexp(
    V: int,
    blank_idx: int,
    block_C: int = 128,
    block_V: int = 128,
    num_stages: int = 2,
    threads: int = 128,
    dtype: TLDtype = torch.float16,
    accum_dtype: TLDtype = torch.float32,
):
    """Online logsumexp of a pre-computed ``X``; see the module docstring for detail.

    Torch Equivalent: ``Denom = X.logsumexp(-1)``
    """
    n_cells = T.dynamic("n_cells")  # packed cells of X
    n_rect = T.dynamic("n_rect")  # B * max_flen * max_glen (rect lbx/lex, flat)
    label_dtype = T.int32
    idx_dtype = T.int64

    v_tail = V % block_V != 0

    @T.prim_func
    def main(
        X: T.Tensor([n_cells, V], dtype),  # packed logits A@W+b
        LabelCell: T.Tensor([n_cells], label_dtype),
        CellIdx: T.Tensor([n_cells], idx_dtype),
        Denom: T.Tensor([n_cells], accum_dtype),
        Lbx: T.Tensor([n_rect], accum_dtype),
        Lex: T.Tensor([n_rect], accum_dtype),
    ):
        with T.Kernel(T.ceildiv(n_cells, block_C), threads=threads) as (bc,):
            x_shared = T.alloc_shared([block_C, block_V], dtype)
            acc_s = T.alloc_fragment([block_C, block_V], accum_dtype)  # X block
            scores_max = T.alloc_fragment([block_C], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_C], accum_dtype)
            scores_scale = T.alloc_fragment([block_C], accum_dtype)
            scores_sum = T.alloc_fragment([block_C], accum_dtype)
            logsum = T.alloc_fragment([block_C], accum_dtype)
            lse = T.alloc_fragment([block_C], accum_dtype)
            masked_logits = T.alloc_fragment([block_C, block_V], accum_dtype)
            block_logit = T.alloc_fragment([block_C], accum_dtype)
            lbx_frag = T.alloc_fragment([block_C], accum_dtype)
            lex_frag = T.alloc_fragment([block_C], accum_dtype)
            lab = T.alloc_fragment([block_C], label_dtype)
            ridx = T.alloc_fragment([block_C], idx_dtype)

            c0 = bc * block_C
            prologue_init(
                c0=c0,
                n_cells=n_cells,
                LabelCell=LabelCell,
                CellIdx=CellIdx,
                lab=lab,
                ridx=ridx,
                lbx_frag=lbx_frag,
                lex_frag=lex_frag,
                logsum=logsum,
                scores_max=scores_max,
                accum_dtype=accum_dtype,
                block_C=block_C,
            )

            for v in T.Pipelined(T.ceildiv(V, block_V), num_stages=num_stages):
                v0 = v * block_V
                if v_tail:
                    for i, j in T.Parallel(block_C, block_V):
                        x_shared[i, j] = X[
                            T.min(c0 + i, n_cells - 1), T.min(v0 + j, V - 1)
                        ]
                else:
                    if c0 + block_C <= n_cells:
                        T.copy(X[c0 : c0 + block_C, v0 : v0 + block_V], x_shared)
                    else:
                        for i, j in T.Parallel(block_C, block_V):
                            x_shared[i, j] = X[T.min(c0 + i, n_cells - 1), v0 + j]
                T.copy(x_shared, acc_s)
                if v_tail:
                    for i, j in T.Parallel(block_C, block_V):
                        if v0 + j >= V:
                            acc_s[i, j] = NEG

                accum_lbx_lex(
                    v0=v0,
                    blank_idx=blank_idx,
                    acc_s=acc_s,
                    lab=lab,
                    masked_logits=masked_logits,
                    block_logit=block_logit,
                    lbx_frag=lbx_frag,
                    lex_frag=lex_frag,
                    block_C=block_C,
                    block_V=block_V,
                )

                update_online_lse(
                    acc_s=acc_s,
                    scores_max=scores_max,
                    scores_max_prev=scores_max_prev,
                    scores_scale=scores_scale,
                    scores_sum=scores_sum,
                    logsum=logsum,
                    block_C=block_C,
                    block_V=block_V,
                )

            epilogue_norm_scatter(
                lse=lse,
                scores_max=scores_max,
                logsum=logsum,
                lbx_frag=lbx_frag,
                lex_frag=lex_frag,
                Denom=Denom,
                Lbx=Lbx,
                Lex=Lex,
                ridx=ridx,
                c0=c0,
                n_cells=n_cells,
                block_C=block_C,
            )

    return main


# ===========================================================================
# Production entry points
# ===========================================================================
@beartype
def proj_lse_run(kernel: Callable):
    def run(A, W, b, label_cell, cell_idx, denom, lbx, lex, X, total):
        kernel(A, W, b, label_cell, cell_idx, denom, lbx.view(-1), lex.view(-1), X)

    return run


@beartype
def logsumexp_run(kernel: Callable):
    def run(X, label_cell, cell_idx, denom, lbx, lex):
        kernel(X, label_cell, cell_idx, denom, lbx.view(-1), lex.view(-1))

    return run


def ref_torch(A, W, Bias, Label=None, blank_idx=None):
    """Torch fp32 reference"""
    joint = torch.einsum("btuh,hk->btuk", A.float(), W.float()) + Bias.float()
    denom = torch.logsumexp(joint, dim=-1)  # (B, T, U)
    if Label is None:
        return denom
    x_blank = joint[..., blank_idx]  # (B, T, U)
    U = joint.shape[2]
    u_idx = torch.arange(U, device=Label.device).clamp(max=Label.shape[1] - 1)
    label_u = Label[:, u_idx]  # (B, U)
    idx = label_u[:, None, :].expand(joint.shape[:-1]).unsqueeze(-1).long()
    x_label = torch.gather(joint, -1, idx).squeeze(-1)  # (B, T, U)
    return denom, x_blank - denom, x_label - denom
