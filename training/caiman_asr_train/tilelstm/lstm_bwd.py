"""
TileLang LSTM backward (reverse-scan) kernels.

All three kernels run the same reverse recurrence over the forward's saved
activations; the per-step math is the recurrent gemm plus ``bwd_gate_grads``
below (t_rev = S-1 .. 0, rW_flat = rW.reshape(4H, H)):

    dh      = delta[t_rev] + dGates[t_rev+1] @ rW_flat  # gemm over 4H, fp32 accum
    c_tanh  = tanh(Cs[t_rev+1])
    dc      = dh * o * (1 - c_tanh^2) + dc_fut
    dI, dF, dG, dO from (dh, dc, As[t_rev], Cs[t_rev])  # see the macro
    dGates[t_rev] = (dI, dF, dG, dO)                    # next step's gemm operand
    dc_fut  = dc * f                                    # carry to step t_rev-1

The cross-block dependency runs through dGates: reverse step t_rev's gemm
reads the full 4H row of dGates[t_rev+1], written at the previous reverse step
by all n_prod = cdiv(H, block_N) column tiles of the m-group. Coop and
persistent order bands with the forward's counter macros; split-K uses
grid-wide barriers instead. The strategies differ in the grid and in where the
dc_fut carry lives.

Coop -- one block per tile (mirrors lstm_coop_fwd); carry in registers:

    grid = (cdiv(H, block_N), cdiv(M, block_M))      # every tile gets a block
    launch cooperative:

        dC_frag = 0                                  # dc_fut carry, registers
        sync_grid()                                  # co-residency for the spins

        for t_rev in S-1 .. 0:                       # SERIAL reverse time
            dh = delta[t_rev]
            if t_rev < S-1:
                band_wait(F, by, ...)                # band stored dGates[t_rev+1]
                dh += dGates[t_rev+1] @ rW_flat
            dGates[t_rev] = gate_grads(dh, dC_frag)  # dC_frag: carry in/out
            band_arrive(F, by)                       # my dGates[t_rev] is out

Split-K -- weight-stationary; carry in global dC; grid barriers, no bands:

    grid = (cdiv(H, block_N), k_splits)                    # block OWNS a K-slice of rW
    launch cooperative:

        rW_res = my K-slice of rW                          # SMEM-resident for ALL steps
        dC[my row share] = 0                               # carry lives in global dC

        for t_rev in S-1 .. 0:
            for every m-tile:                              # phase 1: partial dh
                P[by] = dGates[t_rev+1][:, my K] @ rW_res  # my K-slice, ALL rows
            sync_grid()                                    # all partials visible
            for my red_M-row share:                        # phase 2: reduce + epilogue
                dh = delta[t_rev] + sum_k P[k]             # reduce k_splits partials
                dGates[t_rev] = gate_grads(dh, dC)         # dC: carry in/out
            sync_grid()                                    # dGates[t_rev] out, P free

Persistent -- fixed grid (mirrors lstm_persistent_fwd); carry in global dC:

    grid = blocks_per_sm * num_SMs                             # tile-count independent
    launch cooperative:

        for (by, bx) in my stride of tiles: dC[tile] = 0       # zero-seed the carry
        sync_grid()

        for t_rev in S-1 .. 0:
            for (by, bx) in my stride of tiles:                # many tiles per block
                                                               #   -> no register carry
                if t_rev < S-1: band_wait(F, by, ...)
                dh = delta[t_rev] + dGates[t_rev+1] @ rW_flat  # dGates[S]: zero pad
                dGates[t_rev] = gate_grads(dh, dC[tile])       # dC: re-streamed carry
                band_arrive(F, by)
"""

import math

import tilelang
import tilelang.language as T
import torch
from beartype import beartype
from tilelang.carver.arch import driver
from torch import Tensor

from caiman_asr_train.autotune.dtypes import TLDtype
from caiman_asr_train.tilelstm.lstm_fwd import (
    LSTM_PASS_CONFIGS,
    band_arrive,
    band_wait,
)


@T.macro
def bwd_gate_grads(
    gi,
    gf,
    gg,
    go,
    dh,
    c_prev,
    c_tanh,
    dc_fut,
    dIg,
    dFg,
    dGg,
    dOg,
    dCarry,
    i,
    j,
    dtype_lo,
):
    """
    Shared per-element gate-gradient epilogue of the three backward kernels: from
    this element's loaded operands (the four stored post-activation gates, the
    upstream hidden grad ``dh``, the two neighbouring cells, the carried
    ``dc_fut``), write the four ``dtype_lo`` gate grads at ``[i, j]`` and the
    ``dc * gf`` carry into ``dCarry`` (the caller's ``dc_fut`` source for the
    earlier step). Callers own the surrounding ``T.Parallel`` loop and the operand
    loads, whose layouts genuinely differ (plain global reads, the wpf As
    prefetch, the persistent cp.async staging).

    sigmoid'(x) = x*(1-x), tanh'(x) = 1 - x^2, on the stored activations.
    """
    dO = dh * c_tanh * (go * (1 - go))
    dc = dh * go * (1 - c_tanh * c_tanh) + dc_fut
    dF = dc * c_prev * (gf * (1 - gf))
    dI = dc * gg * (gi * (1 - gi))
    dG = dc * gi * (1 - gg * gg)

    dIg[i, j] = T.cast(dI, dtype_lo)
    dFg[i, j] = T.cast(dF, dtype_lo)
    dGg[i, j] = T.cast(dG, dtype_lo)
    dOg[i, j] = T.cast(dO, dtype_lo)

    # Carry to the earlier step (the caller's dc_fut source).
    dCarry[i, j] = dc * gf


@tilelang.jit(pass_configs=LSTM_PASS_CONFIGS)
@beartype
def lstm_coop_bwd(
    H: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    policy: T.GemmWarpPolicy = T.GemmWarpPolicy.Square,
    wpf: bool = False,
    fpad: int = 1,
    dtype_lo: TLDtype = torch.float16,
    dtype_hi: TLDtype = torch.float32,
):
    """
    Returns an in-place LSTM recurrent-scan BACKWARD kernel.

    Arguments:
        wpf: wait prefetch: prefetch the As operand into SMEM.
        fpad: flags-padding: padding (in multiples of i32)
        policy: the ``T.GemmWarpPolicy`` for the recurrent gemm's warp partition
            (Square for even-16 block_M, FullCol for odd-16 block_M; see
            ``autotune.configs._wmma_ok``).
    """
    assert H % block_K == 0, "H must be divisible by block_K"

    gemm_policy = policy

    # Column blocks per m-group (band members), counting the ragged last tile.
    n_prod = T.cdiv(H, block_N)
    i32 = "int32"  # named: flake8 rejects string literals in TIR annotations

    M = T.dynamic("m")
    S = T.dynamic("s")

    @T.prim_func
    def main(
        As: T.Tensor[(S, M, 4, H), dtype_lo],
        Cs: T.Tensor[(S + 1, M, H), dtype_lo],
        rW: T.Tensor[(4, H, H), dtype_lo],
        delta: T.Tensor[(S, M, H), dtype_lo],
        dGates: T.Tensor[(S + 1, M, 4, H), dtype_lo],
        F: T.Tensor[(T.cdiv(M, block_M), fpad), i32],
    ):
        with T.Kernel(T.cdiv(H, block_N), T.cdiv(M, block_M), threads=threads) as (
            bx,
            by,
        ):
            # MMA
            dG_shared = T.alloc_shared((block_M, block_K), dtype_lo)
            rWb_shared = T.alloc_shared((block_K, block_N), dtype_lo)
            dh_local = T.alloc_fragment((block_M, block_N), dtype_hi)

            if wpf:
                As_shared = T.alloc_shared((4, block_M, block_N), dtype_lo)

            # Temp
            dh_frag = T.alloc_fragment((block_M, block_N), dtype_hi)

            # Carry
            dC_frag = T.alloc_fragment((block_M, block_N), dtype_hi)

            # Per-gate grad fragments, stored coalesced one copy per gate.
            dIg = T.alloc_fragment((block_M, block_N), dtype_lo)
            dFg = T.alloc_fragment((block_M, block_N), dtype_lo)
            dGg = T.alloc_fragment((block_M, block_N), dtype_lo)
            dOg = T.alloc_fragment((block_M, block_N), dtype_lo)

            # ===== Seed =====

            # Thread's index
            tx_v = T.get_thread_binding()

            T.clear(dC_frag)

            # Cooperative anchor: co-residency guarantee for the band spins.
            T.sync_grid()

            # ===== Reverse scan t_rev = S-1 .. 0; every iteration is identical =====

            for t in T.serial(S):
                t_rev = S - 1 - t

                # Ready accumulator
                T.clear(dh_local)

                xN = bx * block_N
                m0 = by * block_M

                m0M = m0 + block_M
                xNN = xN + block_N

                if wpf:
                    # As prefetch
                    for g in T.serial(4):
                        T.async_copy(As[t_rev, m0:m0M, g, xN:xNN], As_shared[g, :, :])

                if t > 0:
                    # Band-local wait: all n_prod column tiles of m-group ``by`` must
                    # have stored dGates[t_rev+1] (the previous reverse step) before this
                    # gemm reads the full 4H row.
                    band_wait(F, by, n_prod * t, tx_v)

                    for kc in T.Pipelined(
                        T.cdiv(4 * H, block_K), num_stages=num_stages
                    ):
                        kK = kc * block_K
                        g = kK // H
                        h0 = kK - g * H
                        h0K = h0 + block_K
                        T.copy(dGates[t_rev + 1, m0:m0M, g, h0:h0K], dG_shared)
                        T.copy(rW[g, h0:h0K, xN:xNN], rWb_shared)
                        # We don't require a policy for correctness but it does effect
                        # valid tile-sizes so we allow consumer to customize to allow
                        # better SM utilization.
                        T.gemm(dG_shared, rWb_shared, dh_local, policy=gemm_policy)

                # Gather the accumulator into a plain fragment
                for i, j in T.Parallel(block_M, block_N):
                    dh_frag[i, j] = dh_local[i, j] + T.cast(
                        delta[t_rev, m0 + i, xN + j], dtype_hi
                    )

                if wpf:
                    # Maybe wait for As
                    T.ptx_wait_group(0)
                    T.sync_threads()

                # Element-wise gate epilogue, inputs straight from global (or the
                # wpf-prefetched As slab).
                for i, j in T.Parallel(block_M, block_N):
                    if wpf:
                        gi = T.cast(As_shared[0, i, j], dtype_hi)
                        gf = T.cast(As_shared[1, i, j], dtype_hi)
                        gg = T.cast(As_shared[2, i, j], dtype_hi)
                        go = T.cast(As_shared[3, i, j], dtype_hi)
                    else:
                        gi = T.cast(As[t_rev, m0 + i, 0, xN + j], dtype_hi)
                        gf = T.cast(As[t_rev, m0 + i, 1, xN + j], dtype_hi)
                        gg = T.cast(As[t_rev, m0 + i, 2, xN + j], dtype_hi)
                        go = T.cast(As[t_rev, m0 + i, 3, xN + j], dtype_hi)

                    dh = dh_frag[i, j]
                    # Cs is stored fp16 (dtype_lo); up-cast each read to fp32 for the
                    # gradient math (the dc_fut carry stays fp32 in dC_frag).
                    c_prev = T.cast(Cs[t_rev, m0 + i, xN + j], dtype_hi)
                    c_tanh = T.tanh(T.cast(Cs[t_rev + 1, m0 + i, xN + j], dtype_hi))
                    dc_fut = dC_frag[i, j]

                    # dC_frag carries dc_fut to step t_rev-1 (same tile, registers).
                    bwd_gate_grads(
                        gi,
                        gf,
                        gg,
                        go,
                        dh,
                        c_prev,
                        c_tanh,
                        dc_fut,
                        dIg,
                        dFg,
                        dGg,
                        dOg,
                        dC_frag,
                        i,
                        j,
                        dtype_lo,
                    )

                # Coalesced store of each gate grad; explicit M/H ranges so the
                # [bM, bN] fragment maps to dGates' (M, H), not the gate axis.
                m1 = m0 + block_M
                xE = xN + block_N
                T.copy(dIg, dGates[t_rev, m0:m1, 0, xN:xE])
                T.copy(dFg, dGates[t_rev, m0:m1, 1, xN:xE])
                T.copy(dGg, dGates[t_rev, m0:m1, 2, xN:xE])
                T.copy(dOg, dGates[t_rev, m0:m1, 3, xN:xE])

                # ARRIVE: this column block's dGates[t_rev] tile for the band is out.
                band_arrive(F, by, tx_v)

    return main


@tilelang.jit(pass_configs=LSTM_PASS_CONFIGS)
@beartype
def lstm_coop_bwd_splitk(
    H: int,
    k_splits: int = 8,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    red_M: int = 8,
    res_kc: int | None = None,
    dtype_lo: TLDtype = torch.float16,
    dtype_hi: TLDtype = torch.float32,
):
    """
    Split-K WEIGHT-STATIONARY recurrent-scan backward

    Arguments:
        k_splits: Number of K-splits (blocks) per column group.
        red_M: the reduction tile height.
        res_kc: the number of K-chunks of the block's resident rW strip.
    """
    assert H % block_K == 0, "H must be divisible by block_K"
    assert (4 * H) % k_splits == 0, "4H must be divisible by k_splits"
    k_range = 4 * H // k_splits  # flat-K extent owned by one block
    assert k_range % block_K == 0, "per-block K range must be divisible by block_K"
    kchunks = k_range // block_K
    if res_kc is None:
        res_kc = kchunks
    assert 1 <= res_kc <= kchunks, "res_kc must be in [1, kchunks]"

    tail = kchunks - res_kc  # K-chunks streamed per m-tile (0 = full strip)

    M = T.dynamic("m")
    S = T.dynamic("s")

    @T.prim_func
    def main(
        As: T.Tensor[(S, M, 4, H), dtype_lo],
        Cs: T.Tensor[(S + 1, M, H), dtype_lo],
        rW: T.Tensor[(4, H, H), dtype_lo],
        delta: T.Tensor[(S, M, H), dtype_lo],
        dGates: T.Tensor[(S + 1, M, 4, H), dtype_lo],
        P: T.Tensor[(k_splits, M, H), dtype_hi],
        dC: T.Tensor[(M, H), dtype_hi],
    ):
        with T.Kernel(T.cdiv(H, block_N), k_splits, threads=threads) as (bx, by):
            # MMA
            rW_res = T.alloc_shared((res_kc, block_K, block_N), dtype_lo)
            dG_shared = T.alloc_shared((block_M, block_K), dtype_lo)
            dh_local = T.alloc_fragment((block_M, block_N), dtype_hi)
            if tail:
                rW_pipe = T.alloc_shared((block_K, block_N), dtype_lo)

            # Phase-2 reducer for one red_M-row tile of this block's row share.
            dh_red = T.alloc_fragment((red_M, block_N), dtype_hi)

            # Epilogue
            dIg = T.alloc_fragment((red_M, block_N), dtype_lo)
            dFg = T.alloc_fragment((red_M, block_N), dtype_lo)
            dGg = T.alloc_fragment((red_M, block_N), dtype_lo)
            dOg = T.alloc_fragment((red_M, block_N), dtype_lo)
            dCo = T.alloc_fragment((red_M, block_N), dtype_hi)

            # ===== Seed =====]

            # Fill resident chunk
            xN = bx * block_N
            for kc in T.serial(res_kc):
                kK = by * k_range + kc * block_K
                g = kK // H
                h0 = kK - g * H
                T.copy(rW[g, h0 : h0 + block_K, xN : xN + block_N], rW_res[kc, :, :])

            rt_tiles = T.cdiv(M, k_splits * red_M)
            share = rt_tiles * red_M

            # Zero-seed the carried cell gradient for this block's OWN row
            # share (the only dC slice it ever reads/writes -> no barrier).
            for rt in T.serial(rt_tiles):
                m0 = by * share + rt * red_M
                for i, j in T.Parallel(red_M, block_N):
                    dC[m0 + i, xN + j] = T.cast(0, dtype_hi)

            # ===== Reverse scan t_rev = S-1 .. 0, two grid-synced phases =====

            for t in T.serial(S):
                t_rev = S - 1 - t

                # Phase 1: partial dh over this block's K range
                for mt in T.serial(T.cdiv(M, block_M)):
                    m0 = mt * block_M
                    m0M = m0 + block_M
                    T.clear(dh_local)

                    # Resident section of mma
                    for kc in T.Pipelined(res_kc, num_stages=num_stages):
                        kK = by * k_range + kc * block_K
                        g = kK // H
                        h0 = kK - g * H
                        h0K = h0 + block_K
                        T.copy(dGates[t_rev + 1, m0:m0M, g, h0:h0K], dG_shared)
                        T.gemm(dG_shared, rW_res[kc, :, :], dh_local)

                    if tail:
                        # Streamed tail of mma
                        for kc2 in T.Pipelined(tail, num_stages=num_stages):
                            kK = by * k_range + (res_kc + kc2) * block_K
                            g = kK // H
                            h0 = kK - g * H
                            h0K = h0 + block_K
                            T.copy(dGates[t_rev + 1, m0:m0M, g, h0:h0K], dG_shared)
                            T.copy(rW[g, h0:h0K, xN : xN + block_N], rW_pipe)
                            T.gemm(dG_shared, rW_pipe, dh_local)

                    T.copy(dh_local, P[by, m0 : m0 + block_M, xN : xN + block_N])

                # Phase-1 -> phase-2 barrier: all partials visible before any block
                # reduces them.
                T.sync_grid()

                # Phase 2: this block reduces its ROW share (the by-th `share`
                # rows of its own bN columns) in red_M-row tiles, summing the
                # k_splits partial layers; then the gate epilogue.
                for rt in T.serial(rt_tiles):
                    m0 = by * share + rt * red_M

                    # First step; fuse in delta-read
                    for i, j in T.Parallel(red_M, block_N):
                        dh_red[i, j] = T.cast(P[0, m0 + i, xN + j], dtype_hi) + T.cast(
                            delta[t_rev, m0 + i, xN + j], dtype_hi
                        )

                    # Other steps, use +=
                    for ks in T.serial(1, k_splits):
                        for i, j in T.Parallel(red_M, block_N):
                            dh_red[i, j] += T.cast(P[ks, m0 + i, xN + j], dtype_hi)

                    # Gate epilogue
                    for i, j in T.Parallel(red_M, block_N):
                        gi = T.cast(As[t_rev, m0 + i, 0, xN + j], dtype_hi)
                        gf = T.cast(As[t_rev, m0 + i, 1, xN + j], dtype_hi)
                        gg = T.cast(As[t_rev, m0 + i, 2, xN + j], dtype_hi)
                        go = T.cast(As[t_rev, m0 + i, 3, xN + j], dtype_hi)

                        dh = dh_red[i, j]
                        c_prev = T.cast(Cs[t_rev, m0 + i, xN + j], dtype_hi)
                        c_tanh = T.tanh(T.cast(Cs[t_rev + 1, m0 + i, xN + j], dtype_hi))
                        dc_fut = dC[m0 + i, xN + j]

                        # dCo carries dc_fut to the earlier step (block-private slice).
                        bwd_gate_grads(
                            gi,
                            gf,
                            gg,
                            go,
                            dh,
                            c_prev,
                            c_tanh,
                            dc_fut,
                            dIg,
                            dFg,
                            dGg,
                            dOg,
                            dCo,
                            i,
                            j,
                            dtype_lo,
                        )

                    m1 = m0 + red_M
                    xE = xN + block_N
                    T.copy(dIg, dGates[t_rev, m0:m1, 0, xN:xE])
                    T.copy(dFg, dGates[t_rev, m0:m1, 1, xN:xE])
                    T.copy(dGg, dGates[t_rev, m0:m1, 2, xN:xE])
                    T.copy(dOg, dGates[t_rev, m0:m1, 3, xN:xE])
                    T.copy(dCo, dC[m0:m1, xN:xE])

                # Phase-2 -> next-step barrier: all of dGates[t_rev] visible before
                # the next step reads it (and all P reads done before phase 1
                # overwrites the slabs).
                T.sync_grid()

    return main


@tilelang.jit(pass_configs=LSTM_PASS_CONFIGS)
@beartype
def lstm_persistent_bwd(
    H: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    blocks_per_sm: int = 1,
    policy: T.GemmWarpPolicy = T.GemmWarpPolicy.Square,
    fpad: int = 1,
    dtype_lo: TLDtype = torch.float16,
    dtype_hi: TLDtype = torch.float32,
):
    """
    Persistent ``T.Persistent`` LSTM recurrent-scan BACKWARD.

    Arguments:
        blocks_per_sm: Grid = blocks_per_sm * SMs
        fpad: flags-padding: padding (in multiples of i32)
        policy: the ``T.GemmWarpPolicy`` for the recurrent gemm's warp partition
            (Square for even-16 block_M, FullCol for odd-16 block_M; see
            ``autotune.configs._wmma_ok``).
    """
    assert H % block_K == 0, "H must be divisible by block_K"
    grid = blocks_per_sm * driver.get_num_sms()
    persist_group = math.gcd(tilelang.cdiv(H, block_N), 8)  # column-tile count (cdiv)
    # Column tiles per m-group (band members): each arrives once on its band counter
    # per completed reverse step (mirrors lstm_persistent_fwd / lstm_coop_bwd).
    n_prod = T.cdiv(H, block_N)
    i32 = "int32"  # named: flake8 rejects string literals in TIR annotations
    M = T.dynamic("m")
    S = T.dynamic("s")

    @T.prim_func
    def main(
        As: T.Tensor[(S, M, 4, H), dtype_lo],
        Cs: T.Tensor[(S + 1, M, H), dtype_lo],
        rW: T.Tensor[(4, H, H), dtype_lo],
        delta: T.Tensor[(S, M, H), dtype_lo],
        dGates: T.Tensor[(S + 1, M, 4, H), dtype_lo],
        dC: T.Tensor[(M, H), dtype_hi],
        F: T.Tensor[(T.cdiv(M, block_M), fpad), i32],
    ):
        with T.Kernel(grid, threads=threads) as block_id:
            # Thread's index, for the band-counter wait/arrive (thread 0 only).
            tx = T.get_thread_binding()
            # MMA
            dG_shared = T.alloc_shared((block_M, block_K), dtype_lo)
            rWb_shared = T.alloc_shared((block_K, block_N), dtype_lo)
            dh_local = T.alloc_fragment((block_M, block_N), dtype_hi)

            dh_shared = T.alloc_shared((block_M, block_N), dtype_hi)

            A_shared = T.alloc_shared((block_M, 4, block_N), dtype_lo)
            Cprev_shared = T.alloc_shared((block_M, block_N), dtype_lo)
            Ccur_shared = T.alloc_shared((block_M, block_N), dtype_lo)

            # Prefetched per-tile carried dc_fut (from global) + the new carry to store.
            dCin_shared = T.alloc_shared((block_M, block_N), dtype_hi)
            dCout_frag = T.alloc_fragment((block_M, block_N), dtype_hi)

            dIg = T.alloc_fragment((block_M, block_N), dtype_lo)
            dFg = T.alloc_fragment((block_M, block_N), dtype_lo)
            dGg = T.alloc_fragment((block_M, block_N), dtype_lo)
            dOg = T.alloc_fragment((block_M, block_N), dtype_lo)

            # ===== Seed: dc_fut = 0 for every tile (last/first-processed step) =====

            for by, bx in T.Persistent(
                [T.cdiv(M, block_M), T.cdiv(H, block_N)],
                grid,
                block_id,
                persist_group,
            ):
                T.clear(dCout_frag)
                T.copy(dCout_frag, dC[by * block_M, bx * block_N])

            T.sync_grid()

            # ===== Reverse scan t_rev = S-1 .. 0 =====
            for t in T.serial(S):
                t_rev = S - 1 - t

                for by, bx in T.Persistent(
                    [T.cdiv(M, block_M), T.cdiv(H, block_N)],
                    grid,
                    block_id,
                    persist_group,
                ):
                    T.clear(dh_local)
                    xN = bx * block_N
                    m0 = by * block_M
                    m0M = m0 + block_M

                    # WAIT: at reverse step t (t > 0) the gemm reads dGates[t_rev+1] over
                    # the full 4H of band ``by`` -- written at reverse step t-1 by all
                    # n_prod column tiles of the band -- so wait until they have arrived.
                    # At t=0, dGates[S] is the caller-zeroed pad (published by the seed
                    # sync_grid), so no wait. The ``bx < n_prod`` in-range guard keeps
                    # surplus over-subscribed blocks out of the band counter (see the
                    # factory docstring); ``bx`` is uniform per block, so the guarded
                    # sync_threads is not divergent.
                    if bx < n_prod:
                        if t > 0:
                            band_wait(F, by, n_prod * t, tx)

                    for kc in T.Pipelined(
                        T.cdiv(4 * H, block_K), num_stages=num_stages
                    ):
                        kK = kc * block_K
                        g = kK // H
                        h0 = kK - g * H
                        h0K = h0 + block_K
                        T.copy(dGates[t_rev + 1, m0:m0M, g, h0:h0K], dG_shared)
                        T.copy(rW[g, h0:h0K, xN : xN + block_N], rWb_shared)
                        # Policy does not change the result, only which (block_M, block_N,
                        # threads) tile the warps can partition (FullCol keeps an odd-16
                        # block_M as m_warp=1 where Square's balance would not); see
                        # lstm_coop_bwd. The epilogue reads the logical [bM, bN] dh, so it
                        # is layout-transparent to the choice.
                        T.gemm(dG_shared, rWb_shared, dh_local, policy=policy)

                    # Drain the gemm's ldmatrix reads of the operand tiles before the
                    # prefetch below overwrites them. The SMEM-merge aliases the
                    # prefetch buffers onto those dead operands, so without this
                    # barrier the cp.async writes clobber operands other warps are
                    # still reading -- a cross-warp WAR (racecheck: cp_async_gs WRITE
                    # vs ptx_ldmatrix_x4 READ).
                    T.sync_threads()

                    T.async_copy(As[t_rev, by * block_M, 0, bx * block_N], A_shared)
                    T.async_copy(Cs[t_rev, by * block_M, bx * block_N], Cprev_shared)
                    T.async_copy(Cs[t_rev + 1, by * block_M, bx * block_N], Ccur_shared)
                    T.async_copy(dC[by * block_M, bx * block_N], dCin_shared)

                    T.copy(dh_local, dh_shared)

                    T.ptx_wait_group(0)
                    T.sync_threads()

                    for i, j in T.Parallel(block_M, block_N):
                        gi = T.cast(A_shared[i, 0, j], dtype_hi)
                        gf = T.cast(A_shared[i, 1, j], dtype_hi)
                        gg = T.cast(A_shared[i, 2, j], dtype_hi)
                        go = T.cast(A_shared[i, 3, j], dtype_hi)

                        dh = dh_shared[i, j] + T.cast(
                            delta[t_rev, by * block_M + i, bx * block_N + j], dtype_hi
                        )
                        c_prev = T.cast(Cprev_shared[i, j], dtype_hi)
                        c_tanh = T.tanh(T.cast(Ccur_shared[i, j], dtype_hi))
                        dc_fut = dCin_shared[i, j]

                        # dCout_frag carries dc_fut to the earlier step (stored to
                        # the tile's global dC slot below).
                        bwd_gate_grads(
                            gi,
                            gf,
                            gg,
                            go,
                            dh,
                            c_prev,
                            c_tanh,
                            dc_fut,
                            dIg,
                            dFg,
                            dGg,
                            dOg,
                            dCout_frag,
                            i,
                            j,
                            dtype_lo,
                        )

                    # Store each gate grad from its fragment
                    bM0 = by * block_M
                    bN0 = bx * block_N
                    bM0M = bM0 + block_M
                    bN0N = bN0 + block_N
                    T.copy(dIg, dGates[t_rev, bM0:bM0M, 0, bN0:bN0N])
                    T.copy(dFg, dGates[t_rev, bM0:bM0M, 1, bN0:bN0N])
                    T.copy(dGg, dGates[t_rev, bM0:bM0M, 2, bN0:bN0N])
                    T.copy(dOg, dGates[t_rev, bM0:bM0M, 3, bN0:bN0N])
                    T.copy(dCout_frag, dC[by * block_M, bx * block_N])

                    # ARRIVE: this tile's dGates[t_rev] is stored (the dc_fut carry in
                    # dC is intra-block, ordered by the fence too). Bump the band counter
                    # so the next reverse step's gemm for this band can read dGates[t_rev]
                    # = dGates[(t_rev-1)+1]. In-range guarded like the wait.
                    if bx < n_prod:
                        band_arrive(F, by, tx)

    return main


@beartype
def lstm_bwd_ref(As: Tensor, Cs: Tensor, R: Tensor, delta: Tensor):
    """
    Pure-PyTorch transcription of ``lstm_coop_bwd``'s reverse scan.

    Takes the *same* inputs the kernel receives -- the saved post-activation
    gates ``As`` ``[S, B, 4, H]`` (gate-major, gate k of unit j at ``As[t,:,k,j]``,
    k = 0..3 -> i, f, g, o), the forward cell states ``Cs`` ``[S+1, B, H]``, the
    recurrent weight ``R`` ``[4, H, H]`` (gate-major) and the upstream
    hidden-output gradient ``delta`` ``[S, B, H]`` (grad of ``Ys[1:]``) -- and
    reproduces the kernel's precision: the recurrent gemm accumulates into fp32
    (``hi``, matching the kernel's fp32 accumulator -- on CUDA via
    ``torch.mm(out_dtype=hi)`` for fp16-in/fp32-accum, on CPU by an explicit
    up-cast), the gate math runs in ``hi``, and ``dGates`` is cast back to
    ``lo = As.dtype`` (the next reverse step's gemm operand). ``hi`` is
    ``promote_types(Cs.dtype, fp32)`` -- fp32 for a fp16/fp32 cell, fp64 only for a
    fp64 cell: the saved cell ``Cs`` is now stored fp16 (``dtype_lo``) but the kernel
    up-casts each read to fp32 for the gradient math (the dc_fut carry, the recurrent
    gemm), so ``hi`` is floored to fp32 rather than following the fp16 storage, and the
    ``Cs[t].to(hi)`` reads below model the kernel's fp16->fp32 up-cast exactly. The
    reverse scan
    is the kernel's per-step body, line for line:

        dh[t]     = delta[t] + (dGates[t+1] @ R_flat)   # R_flat = R.reshape(4H, H)
        c_tanh    = tanh(Cs[t+1])
        dO        = dh * c_tanh * o*(1-o)
        dc        = dh * o * (1 - c_tanh^2) + dc_fut
        dF        = dc * Cs[t] * f*(1-f)
        dI        = dc * g * i*(1-i)
        dG        = dc * i * (1 - g^2)
        dGates[t] = stack(dI, dF, dG, dO)
        dc_fut    = dc * f                              for t = S-1 .. 0, dc_fut[S] = 0

    The recurrent term contracts ``R_flat``'s ``4H`` rows (gate, output-unit) with
    NO transpose, exactly the kernel's no-transpose gemm. Returns ``dGates``
    ``[S, B, H*4]`` reshaped to ``[S, B, 4, H]`` (lo), matching the kernel's
    ``dGates[:S]`` (the kernel pads an extra ``S+1`` slot the reference omits).
    """
    S, B, _, H = As.shape
    lo = As.dtype  # kernel's dtype_lo
    # Compute/accumulator dtype: at least fp32, promoted to fp64 only for a fp64 cell
    # (matching the kernel, which accumulates fp16/fp32 cells in fp32). promote_types
    # floors the now-fp16-STORED Cs back up to fp32 -- the kernel up-casts each Cs read
    # to fp32 for the gradient math, so deriving ``hi`` straight from Cs.dtype would
    # wrongly collapse the dc_fut carry + recurrent gemm to fp16.
    hi = torch.promote_types(Cs.dtype, torch.float32)
    # R_flat[k*H + n, m] = R[k, n, m]: contract (gate k, output-unit n), output m.
    R_flat = R.reshape(4 * H, H)
    dgates = [None] * S
    dc_fut = torch.zeros(B, H, dtype=hi, device=As.device)  # dc_fut[S] = 0
    for t in reversed(range(S)):
        dh = delta[t].to(hi)
        if t < S - 1:
            # Recurrent contribution from the next step's gate gradients. The
            # kernel accumulates this in fp32; mirror with mm(out_dtype=hi) on CUDA
            # (fp16 inputs, fp32 accumulation) and an explicit up-cast on CPU.
            dg_next = dgates[t + 1].reshape(B, 4 * H)
            if dg_next.is_cuda and dg_next.dtype != hi:
                dh = dh + torch.mm(dg_next, R_flat, out_dtype=hi)
            else:
                dh = dh + dg_next.to(hi) @ R_flat.to(hi)
        i = As[t, :, 0].to(hi)
        f = As[t, :, 1].to(hi)
        g = As[t, :, 2].to(hi)
        o = As[t, :, 3].to(hi)
        c_prev = Cs[t].to(hi)
        c_tanh = torch.tanh(Cs[t + 1].to(hi))
        dO = dh * c_tanh * (o * (1 - o))
        dc = dh * o * (1 - c_tanh * c_tanh) + dc_fut
        dF = dc * c_prev * (f * (1 - f))
        dI = dc * g * (i * (1 - i))
        dG = dc * i * (1 - g * g)
        dgates[t] = torch.stack([dI, dF, dG, dO], dim=1).to(lo)  # [B, 4, H]
        dc_fut = dc * f
    return torch.stack(dgates, dim=0)  # [S, B, 4, H]
