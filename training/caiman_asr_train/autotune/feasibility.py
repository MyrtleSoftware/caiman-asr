"""
Kernel-agnostic tile feasibility filters for TileLang tensor-core kernels:
WMMA-atom and warp-partition legality, SMEM fit, and grid-residency checks.

Each kernel composes these with its own shape constraints and per-config SMEM
estimator (see ``tilelstm/autotune/configs.py`` for the worked example).
"""

from collections.abc import Callable

from beartype import beartype
from tilelang.language import GemmWarpPolicy

from caiman_asr_train.autotune.device import DeviceLimits, resident_blocks

# The m16n8k16 MMA atom shape (fp16 operands, fp32 accumulate -- the tensor-core
# instruction every kernel here lowers to).
ATOM_M = 16
ATOM_N = 8
ATOM_K = 16

# Threads per warp -- the CUDA hardware warp width. A legal block has a whole number of
# warps (threads % TPW == 0) and the warp partition is over threads // TPW warps.
TPW = 32


@beartype
def threads_are_warp(cfg: dict) -> bool:
    """
    Check for an integral number of warps (threads % TPW == 0).
    """
    return cfg["threads"] % TPW == 0


@beartype
def mnk_atom_ok(cfg: dict, *, n_block_mult: int) -> bool:
    """
    Whether the tile tiles cleanly into m16n8k16 WMMA atoms.
    """
    return (
        cfg["block_M"] % ATOM_M == 0
        and (n_block_mult * cfg["block_N"]) % ATOM_N == 0
        and cfg["block_K"] % ATOM_K == 0
    )


@beartype
def wmma_ok(cfg: dict, policy: GemmWarpPolicy, *, n_block_mult: int) -> bool:
    """
    Whether ``policy`` gives a LEGAL warp partition for this config's gemm.

    Essentially checking that the output tile (registers) is partitionable into
    warps.
    """
    block_M, block_N, threads = cfg["block_M"], cfg["block_N"], cfg["threads"]

    assert threads % TPW == 0, f"threads {threads} must be a multiple of {TPW}"

    wider_N = n_block_mult * block_N
    num_warps = threads // TPW
    m_warp, n_warp = policy.compute_warp_partition(block_M, wider_N, num_warps)

    return (
        m_warp * n_warp == num_warps
        and block_M % (m_warp * ATOM_M) == 0
        and wider_N % (n_warp * ATOM_N) == 0
    )


@beartype
def regs_per_block(cfg: dict, *, n_block_mult: int, reg_floor_per_thread: int) -> int:
    """
    Per-block register estimate for the residency check: the larger of the caller's
    calibrated optimistic per-thread floor (``reg_floor_per_thread * threads``) and
    the gemm's fp32 output fragment.
    """
    floor = reg_floor_per_thread * cfg["threads"]
    gemms = n_block_mult * cfg["block_M"] * cfg["block_N"]

    return max(floor, gemms)


@beartype
def grid_resident(
    *, grid: int, threads: int, smem: int, lim: DeviceLimits, regs_per_block: int
) -> bool:
    """
    Whether a ``grid``-block cooperative launch co-resides at this SMEM/threads/register
    footprint -- a cooperative launch needs the whole grid resident at once.
    """
    return grid <= resident_blocks(
        lim, smem_per_block=smem, threads=threads, regs_per_block=regs_per_block
    )


@beartype
def smem_fits(cfg: dict, *, smem_fn: Callable, lim: DeviceLimits) -> bool:
    """
    Whether the config's peak per-block SMEM fits one block on the device.
    """
    return smem_fn(cfg) <= lim.smem_per_block
