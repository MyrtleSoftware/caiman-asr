from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import torch
from beartype import beartype

from caiman_asr_train.autotune.bench import tl_log
from caiman_asr_train.autotune.device import device_name
from caiman_asr_train.autotune.dtypes import tl_dtype
from caiman_asr_train.autotune.sweep import autotune_select
from caiman_asr_train.tile_transducer.autotune.configs import (
    logsumexp_configs,
    proj_and_loss_bwd_configs,
    proj_lse_configs,
    transducer_loss_bwd_configs,
)
from caiman_asr_train.tile_transducer.autotune.kernel import Kernel
from caiman_asr_train.tile_transducer.autotune.mock_batch import (
    MOCK_U,
    TuneShape,
    bwd_batch,
    fwd_batch,
    probe_shape,
)
from caiman_asr_train.tile_transducer.tilelang_kernels.tilelang_lse import (
    logsumexp,
    logsumexp_run,
    proj_lse,
    proj_lse_run,
    ref_torch,
)
from caiman_asr_train.tile_transducer.tilelang_kernels.tilelang_transducer import (
    proj_and_loss_bwd,
    proj_and_loss_bwd_run,
    transducer_loss_bwd,
    transducer_loss_bwd_run,
)

N_WARMUP = 3
N_REPEAT = 10
RTOL, ATOL = 2e-2, 2e-2


@beartype
@dataclass(frozen=True)
class _Backend:
    factory: object
    make_configs: Callable
    call: Callable
    stage: Callable
    outputs: Callable
    reference: Callable
    actual: Callable
    tune_finite: bool


@beartype
def _call_proj_lse(kernel, batch: dict) -> None:
    kernel(
        batch["A"],
        batch["W"],
        batch["Bias"],
        batch["label_cell"],
        batch["cell_idx"],
        batch["denom"],
        batch["lbx"],
        batch["lex"],
        batch["X"],
    )


@beartype
def _call_logsumexp(kernel, batch: dict) -> None:
    kernel(
        batch["X"],
        batch["label_cell"],
        batch["cell_idx"],
        batch["denom"],
        batch["lbx"],
        batch["lex"],
    )


@beartype
def _call_transducer_loss_bwd(kernel, batch: dict) -> None:
    kernel(
        batch["c_dense"],
        batch["c_blank"],
        batch["c_label"],
        batch["denom"],
        batch["label_cell"],
        batch["dX"],
    )


@beartype
def _call_proj_and_loss_bwd(kernel, batch: dict) -> None:
    kernel(
        batch["c_dense"],
        batch["c_blank"],
        batch["c_label"],
        batch["denom"],
        batch["label_cell"],
        batch["X"],
        batch["W"],
        batch["dA"],
        batch["db"],
    )


@beartype
def _linear(H: int, V: int, batch: dict, copy_weights: bool) -> torch.nn.Module:
    A = batch["A"]
    lin = torch.nn.Linear(H, V, device=A.device, dtype=A.dtype)
    if copy_weights:
        with torch.no_grad():
            lin.weight.copy_(batch["W"].t())
            lin.bias.copy_(batch["Bias"])
    return lin


@beartype
def _stage_proj_lse(batch: dict, H: int, V: int) -> Callable:
    args = (
        batch["A"],
        batch["W"],
        batch["Bias"],
        batch["label_cell"],
        batch["cell_idx"],
        batch["denom"],
        batch["lbx"],
        batch["lex"],
        batch["X"],
        batch["total"],
    )

    def build(compiled):
        run = proj_lse_run(compiled)
        return lambda: run(*args)

    return build


@beartype
def _stage_logsumexp(batch: dict, H: int, V: int) -> Callable:
    lin = _linear(H, V, batch, copy_weights=True)
    A = batch["A"]
    rest = (
        batch["label_cell"],
        batch["cell_idx"],
        batch["denom"],
        batch["lbx"],
        batch["lex"],
    )

    def build(compiled):
        run = logsumexp_run(compiled)
        return lambda: run(lin(A), *rest)

    return build


@beartype
def _stage_proj_and_loss_bwd(batch: dict, H: int, V: int) -> Callable:
    args = (
        batch["A"],
        batch["W"],
        batch["c_dense"],
        batch["c_blank"],
        batch["c_label"],
        batch["denom"],
        batch["label_cell"],
        batch["X"],
        batch["total"],
    )
    dtype, accum_dtype = batch["A"].dtype, batch["denom"].dtype

    def build(compiled):
        run = proj_and_loss_bwd_run(H, V, compiled, dtype, accum_dtype)
        return lambda: run(*args)

    return build


@beartype
def _stage_transducer_loss_bwd(batch: dict, H: int, V: int) -> Callable:
    lin = _linear(H, V, batch, copy_weights=False)
    A_leaf = batch["A"].detach().requires_grad_(True)
    X_out = lin(A_leaf)
    inputs = [A_leaf, lin.weight, lin.bias]
    dX = batch["dX"]
    args = (
        batch["c_dense"],
        batch["c_blank"],
        batch["c_label"],
        batch["denom"],
        batch["label_cell"],
        dX,
    )

    def build(compiled):
        run = transducer_loss_bwd_run(compiled)

        def stage():
            run(*args)
            torch.autograd.grad(X_out, inputs, grad_outputs=dX, retain_graph=True)

        return stage

    return build


@beartype
def _fwd_reference(batch: dict, H: int, V: int, blank_idx: int) -> tuple:
    A_rect = batch["A"].float().view(1, -1, MOCK_U, H)
    denom_ref, lbx_ref, lex_ref = ref_torch(
        A_rect, batch["W"], batch["Bias"], batch["Label"], blank_idx
    )
    return denom_ref.reshape(-1), lbx_ref.reshape(-1), lex_ref.reshape(-1)


@beartype
def _fwd_actual(batch: dict) -> tuple:
    return batch["denom"], batch["lbx"], batch["lex"]


@beartype
def _bwd_reference(batch: dict, H: int, V: int, blank_idx: int) -> tuple:
    x = batch["X"].float()
    sv = torch.exp(x - batch["denom"][:, None])
    dx = batch["c_dense"][:, None] * sv
    dx[:, blank_idx] -= batch["c_blank"]
    dx.scatter_add_(
        1,
        batch["label_cell"].long()[:, None],
        -batch["c_label"][:, None],
    )
    return (dx,)


@beartype
def _bwd_actual(batch: dict) -> tuple:
    return (batch["dX"],)


_BACKENDS: dict[Kernel, _Backend] = {
    Kernel.proj_lse: _Backend(
        factory=proj_lse,
        make_configs=proj_lse_configs,
        call=_call_proj_lse,
        stage=_stage_proj_lse,
        outputs=lambda b: (b["denom"], b["lbx"], b["lex"], b["X"]),
        reference=_fwd_reference,
        actual=_fwd_actual,
        tune_finite=True,
    ),
    Kernel.logsumexp: _Backend(
        factory=logsumexp,
        make_configs=logsumexp_configs,
        call=_call_logsumexp,
        stage=_stage_logsumexp,
        outputs=lambda b: (b["denom"], b["lbx"], b["lex"]),
        reference=_fwd_reference,
        actual=_fwd_actual,
        tune_finite=True,
    ),
    Kernel.proj_and_loss_bwd: _Backend(
        factory=proj_and_loss_bwd,
        make_configs=proj_and_loss_bwd_configs,
        call=_call_proj_and_loss_bwd,
        stage=_stage_proj_and_loss_bwd,
        outputs=lambda b: (b["X"], b["dA"], b["db"]),
        reference=_bwd_reference,
        actual=lambda b: (b["X"],),
        tune_finite=False,
    ),
    Kernel.transducer_loss_bwd: _Backend(
        factory=transducer_loss_bwd,
        make_configs=transducer_loss_bwd_configs,
        call=_call_transducer_loss_bwd,
        stage=_stage_transducer_loss_bwd,
        outputs=lambda b: (b["dX"],),
        reference=_bwd_reference,
        actual=_bwd_actual,
        tune_finite=False,
    ),
}


@beartype
def _finite(*tensors: torch.Tensor) -> bool:
    return all(bool(torch.isfinite(t).all()) for t in tensors)


@beartype
def make_batch(
    kernel: Kernel,
    shape: TuneShape,
    H: int,
    V: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    accum_dtype: torch.dtype = torch.float32,
) -> dict:
    if kernel.is_forward():
        batch = fwd_batch(shape, H, V, device, dtype=dtype, accum_dtype=accum_dtype)
        if kernel is Kernel.logsumexp:
            batch["X"] = torch.addmm(batch["Bias"], batch["A"], batch["W"])
        return batch

    batch = bwd_batch(shape, H, V, device, dtype=dtype, accum_dtype=accum_dtype)
    n_cells = batch["X"].shape[0]
    batch["dA"] = torch.zeros(n_cells, H, device=device, dtype=dtype)
    batch["db"] = torch.zeros(V, device=device, dtype=accum_dtype)
    if kernel is Kernel.transducer_loss_bwd:
        batch["dX"] = batch["X"].clone()
    return batch


@beartype
def _make_validate(
    kernel: Kernel,
    bk: _Backend,
    tune_batch: dict,
    H: int,
    V: int,
    blank_idx: int,
    device: torch.device,
    dtype: torch.dtype,
    accum_dtype: torch.dtype,
) -> Callable:
    probe = make_batch(
        kernel, probe_shape(), H, V, device, dtype=dtype, accum_dtype=accum_dtype
    )
    ref = bk.reference(probe, H, V, blank_idx)
    # Both backward kernels form dX in place; restore their target from the logits.
    dest = "dX" if kernel is Kernel.transducer_loss_bwd else "X"
    pristine = None if kernel.is_forward() else probe["X"].clone()

    def validate(compiled, cfg) -> None:
        if pristine is not None:
            probe[dest].copy_(pristine)
            probe["db"].zero_()
        bk.call(compiled, probe)
        got = bk.actual(probe)

        assert _finite(*bk.outputs(probe)), f"non-finite probe output: {cfg}"

        if kernel.is_forward():
            denom_ref, lbx_ref, lex_ref = ref
            denom, lbx, lex = got
            torch.testing.assert_close(
                denom.float(), denom_ref.float(), rtol=RTOL, atol=ATOL
            )
            torch.testing.assert_close(
                lbx.float(), lbx_ref.float(), rtol=RTOL, atol=ATOL
            )
        else:
            torch.testing.assert_close(
                got[0].float(), ref[0].float(), rtol=RTOL, atol=ATOL
            )

        if bk.tune_finite:
            assert _finite(*bk.outputs(tune_batch)), f"non-finite at tune shape: {cfg}"

    return validate


@cache
@beartype
def tune(
    kernel: Kernel,
    H: int,
    V: int,
    blank_idx: int,
    device: torch.device,
    shape: TuneShape,
    dtype: torch.dtype = torch.float16,
    accum_dtype: torch.dtype = torch.float32,
) -> tuple[Callable, list[dict]] | None:
    bk = _BACKENDS[kernel]
    configs = bk.make_configs(H, V, blank_idx, device, dtype, accum_dtype)
    tl_log(str(kernel), f"{shape}: {len(configs)} feasible configs")

    tune_batch = make_batch(
        kernel, shape, H, V, device, dtype=dtype, accum_dtype=accum_dtype
    )
    validate = _make_validate(
        kernel, bk, tune_batch, H, V, blank_idx, device, dtype, accum_dtype
    )

    build_stage = bk.stage(tune_batch, H, V)

    return autotune_select(
        factory=bk.factory,
        configs=configs,
        make_thunk=lambda compiled, cfg: build_stage(compiled),
        validate=validate,
        finalize=lambda compiled, cfg: compiled,
        cache_key=(
            device_name(device),
            H,
            shape.cells,
            tl_dtype(dtype),
            tl_dtype(accum_dtype),
            "stage",
            N_WARMUP,
            N_REPEAT,
        ),
        label=str(kernel),
        H=H,
        tune_batch=shape.cells,
        n_warmup=N_WARMUP,
        n_repeat=N_REPEAT,
    )
