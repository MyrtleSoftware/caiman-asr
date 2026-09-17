from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import torch
from beartype import beartype

from caiman_asr_train.autotune.bench import tl_log
from caiman_asr_train.autotune.dtypes import accum_dtype_for, tl_dtype
from caiman_asr_train.tile_transducer.autotune.kernel import Kernel, Mode
from caiman_asr_train.tile_transducer.autotune.mock_batch import (
    TuneShape,
    tune_shape,
)
from caiman_asr_train.tile_transducer.autotune.tuners import _BACKENDS, tune
from caiman_asr_train.tile_transducer.tilelang_kernels.tilelang_lse import (
    logsumexp_run,
    proj_lse_run,
)
from caiman_asr_train.tile_transducer.tilelang_kernels.tilelang_transducer import (
    proj_and_loss_bwd_run,
    transducer_loss_bwd_run,
)
from caiman_asr_train.train_utils.distributed import exec_and_broadcast

_LABEL = "mode"


@beartype
@dataclass(frozen=True)
class TileLangKernels:
    mode: Mode
    fwd_run: Callable
    bwd_run: Callable
    dtype: torch.dtype
    accum_dtype: torch.dtype

    def __str__(self) -> str:
        return f"tilelang ({self.mode}, {self.dtype}/{self.accum_dtype})"


@beartype
def _select_mode(
    H: int,
    V: int,
    blank_idx: int,
    device: torch.device,
    shape: TuneShape,
    dtype: torch.dtype,
    accum_dtype: torch.dtype,
) -> dict:
    """Tune every kernel at ``shape`` and pick the mode with the cheaper fwd+bwd."""
    to_tune = tuple(Kernel)
    tl_log(
        _LABEL,
        f"tuning {len(to_tune)} kernels at {shape} for H={H} V={V} "
        f"{tl_dtype(dtype)}/{tl_dtype(accum_dtype)}",
    )

    configs: dict[Kernel, dict] = {}
    stage_ms: dict[Kernel, float] = {}
    for kern in to_tune:
        tuned = tune(kern, H, V, blank_idx, device, shape, dtype, accum_dtype)
        assert (
            tuned is not None
        ), f"no feasible config for {kern} at {shape} in {tl_dtype(dtype)}"
        _, ranked = tuned
        configs[kern] = ranked[0]["config"]
        stage_ms[kern] = float(ranked[0]["latency_ms"])
        torch.cuda.empty_cache()

    def total_ms(m: Mode) -> float:
        return sum(stage_ms[k] for k in m.pair())

    for m in Mode:
        fwd_kern, bwd_kern = m.pair()
        tl_log(
            _LABEL,
            f"{m} stage: fwd {stage_ms[fwd_kern]:.4f} ms "
            f"+ bwd {stage_ms[bwd_kern]:.4f} ms = {total_ms(m):.4f} ms",
        )

    mode = min(Mode, key=total_ms)
    fwd_kern, bwd_kern = mode.pair()
    tl_log(_LABEL, f"winner: {mode}")
    return dict(mode=mode.value, fwd_cfg=configs[fwd_kern], bwd_cfg=configs[bwd_kern])


@cache
@beartype
def select_tilelang_kernels(
    H: int,
    V: int,
    blank_idx: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> TileLangKernels:
    assert dtype is not torch.float64, (
        "the fused TileLang joint+loss backward cannot take an fp64 gemm operand, so "
        "fp64 cannot be mode-selected; build the unfused kernels directly, or use "
        "--transducer_backend cuda/triton"
    )
    accum_dtype = accum_dtype_for(dtype)
    choice = exec_and_broadcast(
        _select_mode, H, V, blank_idx, device, tune_shape(), dtype, accum_dtype
    )

    mode = Mode(choice["mode"])
    fwd_kern, bwd_kern = mode.pair()
    fwd = _BACKENDS[fwd_kern].factory(**choice["fwd_cfg"])
    bwd = _BACKENDS[bwd_kern].factory(**choice["bwd_cfg"])
    common = dict(mode=mode, dtype=dtype, accum_dtype=accum_dtype)

    if mode is Mode.fused:
        return TileLangKernels(
            fwd_run=proj_lse_run(fwd),
            bwd_run=proj_and_loss_bwd_run(H, V, bwd, dtype, accum_dtype),
            **common,
        )
    return TileLangKernels(
        fwd_run=logsumexp_run(fwd),
        bwd_run=transducer_loss_bwd_run(bwd),
        **common,
    )
