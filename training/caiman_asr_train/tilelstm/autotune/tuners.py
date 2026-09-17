"""
Per-backend autotuner entry points for the TileLang LSTM kernels.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import torch
from beartype import beartype

from caiman_asr_train.autotune.bench import n_repeat_for
from caiman_asr_train.autotune.device import device_name
from caiman_asr_train.autotune.dtypes import tl_dtype
from caiman_asr_train.autotune.sweep import autotune_select
from caiman_asr_train.tilelstm.autotune.buffers import bwd_buffers, scan_buffers
from caiman_asr_train.tilelstm.autotune.configs import (
    coop_bwd_configs,
    coop_fwd_configs,
    persistent_bwd_configs,
    persistent_fwd_configs,
    splitk_bwd_configs,
)
from caiman_asr_train.tilelstm.autotune.kernel import Family, Kernel
from caiman_asr_train.tilelstm.autotune.shapes import PROFILE_SEQ
from caiman_asr_train.tilelstm.autotune.wrappers import (
    wrap_coop_bwd_flags,
    wrap_coop_fwd_flags,
    wrap_splitk_bwd,
)
from caiman_asr_train.tilelstm.lstm_bwd import (
    lstm_bwd_ref,
    lstm_coop_bwd,
    lstm_coop_bwd_splitk,
    lstm_persistent_bwd,
)
from caiman_asr_train.tilelstm.lstm_fwd import (
    lstm_coop_fwd,
    lstm_persistent_fwd,
    lstm_ref,
)


@beartype
@dataclass(frozen=True)
class _Backend:
    """
    Everything ``tune`` / ``bench`` need to drive one backend through ``autotune_select``.
    """

    factory: object
    make_configs: Callable
    wrap: Callable


@beartype
def _wrap_config(fn: Callable, **kwargs) -> Callable:
    """
    Build a backend's ``make_configs(H, B, device, lo, hi)`` from its generator ``fn``.
    """
    return lambda H, B, device, lo, hi: fn(
        H, B, device, tl_dtype(lo), tl_dtype(hi), **kwargs
    )


_BACKENDS: dict[Kernel, _Backend] = {
    Kernel.fwd_c: _Backend(
        lstm_coop_fwd,
        _wrap_config(coop_fwd_configs),
        wrap_coop_fwd_flags,
    ),
    Kernel.fwd_p: _Backend(
        lstm_persistent_fwd,
        _wrap_config(persistent_fwd_configs),
        wrap_coop_fwd_flags,
    ),
    Kernel.fwd_ct: _Backend(
        lstm_coop_fwd,
        _wrap_config(coop_fwd_configs, save_gates=True),
        wrap_coop_fwd_flags,
    ),
    Kernel.fwd_pt: _Backend(
        lstm_persistent_fwd,
        _wrap_config(persistent_fwd_configs, save_gates=True),
        wrap_coop_fwd_flags,
    ),
    Kernel.bwd_c: _Backend(
        lstm_coop_bwd,
        _wrap_config(coop_bwd_configs),
        wrap_coop_bwd_flags,
    ),
    Kernel.bwd_s: _Backend(
        lstm_coop_bwd_splitk,
        _wrap_config(splitk_bwd_configs),
        wrap_splitk_bwd,
    ),
    Kernel.bwd_p: _Backend(
        lstm_persistent_bwd,
        _wrap_config(persistent_bwd_configs),
        wrap_coop_bwd_flags,
    ),
}


# ---------------------------------------------------------------------------
# Per-family call contract: buffers, the call-args order, and validation.
# ---------------------------------------------------------------------------


@beartype
def make_call_args(
    family: Family, H: int, B: int, S: int, lo: torch.dtype, hi: torch.dtype, device
) -> tuple:
    """
    The kernel's positional call args (in-place buffers) for ``family``.
    """
    if family is Family.bwd:
        return bwd_buffers(H, B, S, lo, hi, device)
    return scan_buffers(H, B, S, lo, hi, device)


@beartype
def _make_validate(
    family: Family, call_args: tuple, H: int, B: int, S: int, lo, hi, device
) -> Callable:
    """
    A per-config ``validate(wrapped_call)`` that runs the kernel on FRESH
    output buffers and ``assert_close``s against the reference (computed once).
    """
    if family is Family.bwd:
        As, Cs, R, delta, dGates = call_args
        ref = lstm_bwd_ref(As, Cs, R, delta)

        def validate(call: Callable) -> None:
            v_dgates = torch.zeros_like(dGates)
            call(As, Cs, R, delta, v_dgates)
            torch.testing.assert_close(v_dgates[:S], ref, atol=1e-2, rtol=1e-2)

        return validate

    Y, C0, R, gates = call_args[:4]
    ref_ys, ref_cs = lstm_ref(Y, C0, R, gates)
    save_gates = family is Family.fwd_train
    # The gate-saving forward overwrites the gates buffer in place (As aliases Gs), so
    # each config must validate on a PRISTINE copy -- otherwise the 2nd+ config would read
    # the 1st config's activations as its projection input. (The bench thunk's reuse of
    # the shared ``gates`` is harmless: latency is content-invariant.) Snapshot here,
    # before any call mutates ``gates``.
    gates_pristine = gates.clone() if save_gates else gates

    def validate(call: Callable) -> None:
        val_ys, val_cs = scan_buffers(H, B, S, lo, hi, device)[4:]
        g = gates_pristine.clone() if save_gates else gates
        call(Y, C0, R, g, val_ys, val_cs)
        torch.testing.assert_close(val_ys, ref_ys, atol=1e-2, rtol=1e-2)
        # Cs is stored fp16; compare against the fp32 reference down-cast to fp16.
        torch.testing.assert_close(
            val_cs, ref_cs.to(val_cs.dtype), atol=1e-2, rtol=1e-2
        )

    return validate


@cache
@beartype
def tune(
    backend: Kernel,
    H: int,
    device: torch.device,
    tune_batch: int,
    lo: torch.dtype,
    hi: torch.dtype,
    config_ranker: Callable | None = None,
    n_warmup: int = 3,
    n_repeat: int | None = None,
) -> tuple[Callable, list[dict]] | None:
    """
    Autotune ``backend`` (any of the seven in ``_BACKENDS``) for ``(H, dtypes)`` at
    ``tune_batch`` and return ``(kernel, ranked)`` -- the validated winner kernel
    (dynamic in M and S) and ``ranked``: every feasible config with its measured
    tune-batch latency, as ``{"config", "latency_ms"}`` dicts sorted ascending
    (``ranked[0]`` is the winner).
    """
    bk = _BACKENDS[backend]
    B, S = tune_batch, PROFILE_SEQ
    if n_repeat is None:
        n_repeat = n_repeat_for(tune_batch)
    call_args = make_call_args(backend.family, H, B, S, lo, hi, device)
    fam_validate = _make_validate(backend.family, call_args, H, B, S, lo, hi, device)

    # Bind the wrapped call once (the bench reuses the thunk)
    def make_thunk(kernel, cfg):
        return lambda call=bk.wrap(kernel, cfg): call(*call_args)

    def validate(kernel, cfg):
        fam_validate(bk.wrap(kernel, cfg))

    # Prune the candidate set BEFORE handing it to autotune_select (which elaborates every
    # config it gets): the tree ranker keeps the top-K predicted-fastest, so only ~K
    # configs are elaborated. The cache_key is just (device, tune_batch) -- everything else
    # that affects the timing (backend, H, dtypes, kernel body) is captured by each config's
    # TIR hash, which names its bench-cache file.
    configs = bk.make_configs(H, B, device, lo, hi)
    if config_ranker is not None:
        configs = config_ranker(configs, backend, H, tune_batch)

    # Inject bench-info into cache key
    cache_key = (device_name(device), tune_batch, "lean", n_warmup, n_repeat)

    return autotune_select(
        factory=bk.factory,
        configs=configs,
        make_thunk=make_thunk,
        validate=validate,
        finalize=bk.wrap,
        cache_key=cache_key,
        label=backend,
        H=H,
        tune_batch=tune_batch,
        n_warmup=n_warmup,
        n_repeat=n_repeat,
    )
