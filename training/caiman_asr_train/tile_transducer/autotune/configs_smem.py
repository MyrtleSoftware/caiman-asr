"""
Peak per-block SMEM, in bytes, for each TileLang transducer kernel -- the single
source of truth the ``smem_fits`` filters in ``configs.py`` bind as ``smem_fn``.
"""

from beartype import beartype

from caiman_asr_train.autotune.dtypes import dtype_bytes


@beartype
def proj_lse_smem_bytes(cfg: dict) -> int:
    bC, bV, bH, ns = cfg["block_C"], cfg["block_V"], cfg["block_H"], cfg["num_stages"]
    return dtype_bytes(cfg, "dtype") * (ns * (bC * bH + bH * bV) + bV)


@beartype
def logsumexp_smem_bytes(cfg: dict) -> int:
    bC, bV, ns = cfg["block_C"], cfg["block_V"], cfg["num_stages"]
    return dtype_bytes(cfg, "dtype") * ns * bC * bV


@beartype
def proj_and_loss_bwd_smem_bytes(cfg: dict) -> int:
    bC, bV, bH, ns = cfg["block_C"], cfg["block_V"], cfg["block_H"], cfg["num_stages"]
    return dtype_bytes(cfg, "dtype") * ns * (bC * bV + bH * bV)


@beartype
def transducer_loss_bwd_smem_bytes(cfg: dict) -> int:
    return dtype_bytes(cfg, "dtype") * cfg["block_C"] * cfg["block_V"]
