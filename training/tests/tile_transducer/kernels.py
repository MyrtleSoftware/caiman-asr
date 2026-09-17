"""Untuned ``TileLangKernels`` for tests."""

import torch
from beartype import beartype

from caiman_asr_train.autotune.dtypes import accum_dtype_for
from caiman_asr_train.tile_transducer.autotune import Mode, TileLangKernels
from caiman_asr_train.tile_transducer.tilelang_kernels.tilelang_lse import (
    logsumexp,
    logsumexp_run,
    proj_lse,
    proj_lse_run,
)
from caiman_asr_train.tile_transducer.tilelang_kernels.tilelang_transducer import (
    proj_and_loss_bwd,
    proj_and_loss_bwd_run,
    transducer_loss_bwd,
    transducer_loss_bwd_run,
)


@beartype
def untuned_kernels(
    mode: Mode,
    H: int,
    V: int,
    blank_idx: int,
    dtype: torch.dtype = torch.float16,
    fwd_cfg: dict | None = None,
    bwd_cfg: dict | None = None,
) -> TileLangKernels:
    """The kernel pair for ``mode``, JIT-compiled at ``fwd_cfg``/``bwd_cfg`` (or the
    kernel defaults). ``H`` is unused in the unfused mode, which does not project."""
    accum_dtype = accum_dtype_for(dtype)
    dts = dict(dtype=dtype, accum_dtype=accum_dtype)
    fwd_kw = {**(fwd_cfg or {}), **dts}
    bwd_kw = {**(bwd_cfg or {}), **dts}
    common = dict(mode=mode, **dts)

    if mode is Mode.fused:
        bwd = proj_and_loss_bwd(H, V, blank_idx, **bwd_kw)
        return TileLangKernels(
            fwd_run=proj_lse_run(proj_lse(H, V, blank_idx, **fwd_kw)),
            bwd_run=proj_and_loss_bwd_run(H, V, bwd, dtype, accum_dtype),
            **common,
        )
    return TileLangKernels(
        fwd_run=logsumexp_run(logsumexp(V, blank_idx, **fwd_kw)),
        bwd_run=transducer_loss_bwd_run(transducer_loss_bwd(V, blank_idx, **bwd_kw)),
        **common,
    )
