"""
Feasible tile sets for the TileLang transducer kernels.

Each public ``*_configs`` is a generator + filter chain: the raw cross-product from
``candidates.py``, then the static shape / warp-partition / accumulator-budget
constraints, then the launch constraint.
"""

from functools import partial

import torch
from beartype import beartype
from tilelang.language import GemmWarpPolicy

from caiman_asr_train.autotune.device import device_limits
from caiman_asr_train.autotune.dtypes import dtype_bytes, tl_dtype
from caiman_asr_train.autotune.feasibility import smem_fits, threads_are_warp, wmma_ok
from caiman_asr_train.tile_transducer.autotune.candidates import (
    logsumexp_candidate_tiles,
    proj_and_loss_bwd_candidate_tiles,
    proj_lse_candidate_tiles,
    transducer_loss_bwd_candidate_tiles,
)
from caiman_asr_train.tile_transducer.autotune.configs_smem import (
    logsumexp_smem_bytes,
    proj_and_loss_bwd_smem_bytes,
    proj_lse_smem_bytes,
    transducer_loss_bwd_smem_bytes,
)

# Accumulator-register budgets, in 4-byte accumulator slots per thread.
ACC_REGS_LSE = 256
ACC_REGS_FUSED_BWD = 240


@beartype
def _dtypes(dtype: torch.dtype, accum_dtype: torch.dtype) -> dict:
    return dict(dtype=tl_dtype(dtype), accum_dtype=tl_dtype(accum_dtype))


@beartype
def _acc_cv_ok(cfg: dict) -> bool:
    """
    Whether the [block_C, block_V] accumulator fits the per-thread register budget.
    """
    elems = cfg["block_C"] * cfg["block_V"]
    return elems * dtype_bytes(cfg, "accum_dtype") <= 4 * ACC_REGS_LSE * cfg["threads"]


@beartype
def _acc_fused_bwd_ok(cfg: dict) -> bool:
    bC, bV, bH = cfg["block_C"], cfg["block_V"], cfg["block_H"]
    lo, hi = dtype_bytes(cfg, "dtype"), dtype_bytes(cfg, "accum_dtype")
    return bC * (bV + bH) * (hi + lo) <= 4 * ACC_REGS_FUSED_BWD * cfg["threads"]


@beartype
def _warps_partition(cfg: dict, *, cols: str) -> bool:
    """
    Whether the gemm's [block_C, cfg[cols]] output tile admits a legal FullRow warp
    partition.
    """
    tile = dict(block_M=cfg["block_C"], block_N=cfg[cols], threads=cfg["threads"])
    return wmma_ok(tile, GemmWarpPolicy.FullRow, n_block_mult=1)


@beartype
def proj_lse_configs(
    H: int,
    V: int,
    blank_idx: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    accum_dtype: torch.dtype = torch.float32,
) -> list[dict]:
    lim = device_limits(device)
    base = dict(H=H, V=V, blank_idx=blank_idx, **_dtypes(dtype, accum_dtype))

    # Candidates
    configs = proj_lse_candidate_tiles(base)

    # Static constraints
    configs = filter(threads_are_warp, configs)
    configs = filter(partial(_warps_partition, cols="block_V"), configs)
    configs = filter(_acc_cv_ok, configs)
    # Launch constraints
    configs = filter(partial(smem_fits, smem_fn=proj_lse_smem_bytes, lim=lim), configs)

    return list(configs)


@beartype
def logsumexp_configs(
    H: int,
    V: int,
    blank_idx: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    accum_dtype: torch.dtype = torch.float32,
) -> list[dict]:
    lim = device_limits(device)
    base = dict(V=V, blank_idx=blank_idx, **_dtypes(dtype, accum_dtype))

    # Candidates
    configs = logsumexp_candidate_tiles(base)

    # Static constraints
    configs = filter(threads_are_warp, configs)
    configs = filter(_acc_cv_ok, configs)
    # Launch constraints
    configs = filter(partial(smem_fits, smem_fn=logsumexp_smem_bytes, lim=lim), configs)

    return list(configs)


@beartype
def proj_and_loss_bwd_configs(
    H: int,
    V: int,
    blank_idx: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    accum_dtype: torch.dtype = torch.float32,
) -> list[dict]:
    lim = device_limits(device)
    base = dict(H=H, V=V, blank_idx=blank_idx, **_dtypes(dtype, accum_dtype))

    # Candidates
    configs = proj_and_loss_bwd_candidate_tiles(base)

    # Static constraints
    configs = filter(threads_are_warp, configs)
    configs = filter(partial(_warps_partition, cols="block_H"), configs)
    configs = filter(_acc_fused_bwd_ok, configs)
    # Launch constraints
    configs = filter(
        partial(smem_fits, smem_fn=proj_and_loss_bwd_smem_bytes, lim=lim), configs
    )

    return list(configs)


@beartype
def transducer_loss_bwd_configs(
    H: int,
    V: int,
    blank_idx: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    accum_dtype: torch.dtype = torch.float32,
) -> list[dict]:
    lim = device_limits(device)
    base = dict(V=V, blank_idx=blank_idx, **_dtypes(dtype, accum_dtype))

    # Candidates
    configs = transducer_loss_bwd_candidate_tiles(base)

    # Static constraints
    configs = filter(threads_are_warp, configs)
    configs = filter(_acc_cv_ok, configs)
    # Launch constraints
    configs = filter(
        partial(smem_fits, smem_fn=transducer_loss_bwd_smem_bytes, lim=lim), configs
    )

    return list(configs)
