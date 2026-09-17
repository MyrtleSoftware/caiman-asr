"""
Tests for the kernel-agnostic tile feasibility filters (``mnk_atom_ok`` /
``wmma_ok``), pinned to the exact formulas they replaced so drift cannot
silently flip a verdict.

All CUDA-free (pure functions), so it runs anywhere. These tests pin the
generic ``caiman_asr_train.autotune`` package alone and must not import
``caiman_asr_train.tilelstm``.
"""

import pytest
from tilelang.language import GemmWarpPolicy

from caiman_asr_train.autotune.feasibility import mnk_atom_ok, wmma_ok


def test_mnk_atom_ok_is_n_block_mult_aware():
    # The N atom (8) is checked on the REAL gemm width n_block_mult*block_N. block_N=4 is
    # NOT a clean n8 atom for a [block_M, block_N] gemm (mult=1) but IS for the gate-major
    # [block_M, 4*block_N] gemm (mult=4: 16 % 8 == 0). block_M/block_K are mult-independent.
    cfg = {"block_M": 64, "block_N": 4, "block_K": 64}
    assert mnk_atom_ok(cfg, n_block_mult=1) is False
    assert mnk_atom_ok(cfg, n_block_mult=4) is True


def test_mnk_atom_ok_requires_n_block_mult():
    # Non-defaulted, like wmma_ok: a call site cannot silently assume the block_N width.
    with pytest.raises(TypeError):
        mnk_atom_ok({"block_M": 64, "block_N": 16, "block_K": 64})


# GemmWarpPolicy warp-partition legality (doomed-compile prune) for ALL THREE policies:
# FullRow (fwd), FullCol (coop_bwd odd-16 bM), Square (persistent_bwd / split-K / even-16
# coop_bwd). Cases cover the simple legal/illegal paths AND each branch's fallback (legal
# + illegal). Replica of GemmWarpPolicy.compute_warp_partition; matches the campaign's
# full sm_89/sm_120 enumerations (0 compiled config excluded, every warp_partition caught
# across all three policies).
@pytest.mark.parametrize(
    "block_M, block_N, threads, policy, legal",
    [
        # FullRow: n_warp=1, fallback splits warps to N
        (64, 16, 128, GemmWarpPolicy.FullRow, True),  # 64/4=16 rows -> %16 ok
        (
            96,
            16,
            128,
            GemmWarpPolicy.FullRow,
            False,
        ),  # 96/4=24 -> warp_row_tiles%16 != 0
        (128, 32, 256, GemmWarpPolicy.FullRow, True),  # 128/8=16 ok
        (48, 32, 256, GemmWarpPolicy.FullRow, False),  # fallback m=3,n=2 -> m*n=6 != 8
        (
            64,
            32,
            256,
            GemmWarpPolicy.FullRow,
            True,
        ),  # fallback m=4,n=2 -> all divisible
        (32, 8, 128, GemmWarpPolicy.FullRow, False),  # fallback -> block_N%(2*8) != 0
        # FullCol: m_warp=1, fallback splits warps to M
        (32, 64, 128, GemmWarpPolicy.FullCol, True),  # m=1,n=4 -> 64%32, 32%16 ok
        (32, 16, 128, GemmWarpPolicy.FullCol, True),  # fallback n=2,m=2 -> legal
        (
            32,
            16,
            256,
            GemmWarpPolicy.FullCol,
            False,
        ),  # fallback n=2,m=4 -> bM%(4*16)!=0
        # Square: balanced split
        (64, 64, 128, GemmWarpPolicy.Square, True),  # balanced m=2,n=2
        (32, 32, 128, GemmWarpPolicy.Square, True),  # balanced m=2,n=2
        (48, 32, 256, GemmWarpPolicy.Square, False),  # balanced m=2,n=4 -> 48%(2*16)!=0
    ],
)
def test_wmma_ok(block_M, block_N, threads, policy, legal):
    # n_block_mult=1: these cases pin the partition for a [block_M, block_N] gemm
    # (the backward/split-K column width).
    cfg = {"block_M": block_M, "block_N": block_N, "threads": threads}
    assert wmma_ok(cfg, policy, n_block_mult=1) is legal


def test_wmma_ok_requires_n_block_mult():
    # The multiplier is non-defaulted: omitting it is a TypeError, so no call site can
    # silently assume the [block_M, block_N] width for the gate-major forward gemm.
    cfg = {"block_M": 64, "block_N": 16, "threads": 128}
    with pytest.raises(TypeError):
        wmma_ok(cfg, GemmWarpPolicy.FullRow)


@pytest.mark.parametrize("block_M, block_N", [(64, 8), (96, 32), (128, 64)])
def test_wmma_ok_fullrow_quad_noop_at_prod_threads(block_M, block_N):
    # At the production fwd thread count (128), FullRow on block_M in {64,96,128} puts
    # n_warp=1 (every warp owns the full column range so the gate epilogue can gather all
    # 4 gates onto one lane), so the gate-major 4*block_N width is a no-op vs block_N: the
    # verdict is identical at n_block_mult 4 and 1. This is exactly why the production fwd
    # grid (threads=128) is unchanged by the quad fix. A spread of block_N pins it.
    cfg = {"block_M": block_M, "block_N": block_N, "threads": 128}
    p = GemmWarpPolicy.FullRow
    assert wmma_ok(cfg, p, n_block_mult=4) == wmma_ok(cfg, p, n_block_mult=1)


def test_wmma_ok_quad_multiplier_is_not_vacuous():
    # threads=256 + block_M=64 forces FullRow to fall back to n_warp>1, where the real
    # 4*block_N column width flips the warp-legality vs assuming block_N -- proving the
    # multiplier is actually used. (Such tiles are epilogue-gated out of the fwd grid, so
    # this is not a production change, only a guard that the kwarg threads through.)
    cfg = {"block_M": 64, "block_N": 8, "threads": 256}
    p = GemmWarpPolicy.FullRow
    assert wmma_ok(cfg, p, n_block_mult=4) != wmma_ok(cfg, p, n_block_mult=1)
