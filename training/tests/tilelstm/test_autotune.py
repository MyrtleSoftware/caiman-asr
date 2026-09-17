"""
Tests for the TileLang LSTM autotuner / kernel selection.

SM80-gated: tuning builds and benchmarks real kernels (cp.async needs SM80+). Kept
to one small shape so the autotune (a handful of configs across both backends) stays
fast. ``select_fastest`` builds the tree config-ranker (bundle load-or-train + a host
GPU spec probe) before ``tune``; tests that don't exercise the ranker stub it out with
``identity_ranker`` so they are hermetic (CPU-safe, and independent of test order /
the process-level ranker caches).
"""

from unittest.mock import patch

import pytest
import torch
from _kernels import (
    assert_bwd_close,
    assert_fwd_close,
    requires_tile_kernels,
    scan_inputs,
)

from caiman_asr_train.tilelstm.autotune import (
    select_bwd_train_kernel,
    select_fwd_kernel,
    select_fwd_train_kernel,
)
from caiman_asr_train.tilelstm.autotune.buffers import bwd_buffers
from caiman_asr_train.tilelstm.autotune.configs import (
    coop_bwd_configs,
    coop_fwd_configs,
    persistent_bwd_configs,
    persistent_fwd_configs,
)
from caiman_asr_train.tilelstm.autotune.select import select_fastest
from caiman_asr_train.tilelstm.lstm_bwd import lstm_bwd_ref
from caiman_asr_train.tilelstm.lstm_fwd import lstm_ref


@pytest.fixture
def identity_ranker(monkeypatch):
    """
    Stub ``select``'s tree config-ranker with an identity passthrough.

    ``select_fastest`` builds the ranker before calling ``tune``; the real one loads (or
    trains) the model bundle and probes the host GPU, both memoised process-wide. Tests
    that don't exercise the ranker use this stub so they are hermetic: CPU-safe, no
    bundle build inside a redirected cache dir, and no dependence on what earlier tests
    left in ``make_tree_ranker``'s / ``ensure_bundle``'s caches (order independence).
    """
    from caiman_asr_train.tilelstm.autotune import select

    monkeypatch.setattr(
        select,
        "make_tree_ranker",
        lambda **kwargs: lambda configs, backend, H, tune_batch: configs,
    )


@requires_tile_kernels
@pytest.mark.parametrize(
    "gen",
    [
        coop_fwd_configs,
        persistent_fwd_configs,
        coop_bwd_configs,
        persistent_bwd_configs,
    ],
)
def test_config_generators_yield_feasible_configs(gen):
    """
    Each generator yields at least one config on this GPU, and every config
    respects the kernel's tile contract: WMMA-atom multiples (block_M % 16,
    block_N % 8, block_K % 16), block_K | H, and block_N <= H. Note ragged-N: H
    need NOT divide block_N (the forward grid now carries non-dividing block_N
    {24, 48}); the kernel predicates the ragged last column tile. This is the
    behavioral coverage of the feasibility filter (residency/SMEM/warp legality).
    """
    device = torch.device("cuda")
    H, B = 512, 64
    configs = gen(H, B, device, "float16", "float32")
    assert configs, "expected at least one feasible config"
    for cfg in configs:
        assert cfg["block_M"] % 16 == 0 and cfg["block_N"] % 8 == 0
        assert cfg["block_K"] % 16 == 0 and H % cfg["block_K"] == 0
        assert cfg["block_N"] <= H
        assert cfg["threads"] in (128, 256)


@requires_tile_kernels
def test_select_fwd_kernel_matches_ref(cap_autotune_configs):
    """
    select_fwd_kernel returns a tuned kernel that runs the scan in place and
    matches the PyTorch reference.
    """
    device = torch.device("cuda")
    H, B, S = 64, 64, 6
    Y, C0, R, gates, Ys, Cs = scan_inputs(H, B, S, device)

    kernel = select_fwd_kernel(H, B, device, torch.float16, torch.float32)
    kernel(Y, C0, R, gates, Ys, Cs)

    ref_ys, ref_cs = lstm_ref(Y, C0, R, gates)
    assert_fwd_close(Ys, Cs, ref_ys, ref_cs)


@requires_tile_kernels
def test_select_fwd_train_kernel_matches_ref(cap_autotune_configs):
    """
    select_fwd_train_kernel returns a tuned gate-saving kernel (6-arg call contract;
    it overwrites ``gates`` in place with the activations) whose Ys/Cs match the
    reference. The reference is built from a pristine clone (the kernel mutates gates).
    """
    device = torch.device("cuda")
    H, B, S = 64, 64, 6
    Y, C0, R, gates, Ys, Cs = scan_inputs(H, B, S, device)
    gates_in = gates.clone()  # the gate-saving kernel overwrites `gates` in place

    kernel = select_fwd_train_kernel(H, B, device, torch.float16, torch.float32)
    kernel(Y, C0, R, gates, Ys, Cs)

    ref_ys, ref_cs = lstm_ref(Y, C0, R, gates_in)
    assert_fwd_close(Ys, Cs, ref_ys, ref_cs)


@requires_tile_kernels
def test_select_bwd_train_kernel_matches_ref(cap_autotune_configs):
    """
    select_bwd_train_kernel returns a tuned backward kernel whose dGates match the
    reference (comparing the kernel's S+1-slot buffer against the S-slot ref).
    """
    device = torch.device("cuda")
    H, B, S = 64, 64, 6
    As, Cs, R, delta, dGates = bwd_buffers(
        H, B, S, torch.float16, torch.float32, device
    )

    kernel = select_bwd_train_kernel(H, B, device, torch.float16, torch.float32)
    kernel(As, Cs, R, delta, dGates)

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


def test_select_fwd_kernel_propagates_correctness_failure(identity_ranker):
    """
    An AssertionError from assert_close inside tune must propagate out of
    select_fwd_kernel -- select has no try/except around tune, so it never swallows it.
    """
    device = torch.device("cpu")  # no GPU needed -- tune and the ranker are stubbed

    def _tune_raises_assertion(*args, **kwargs):
        raise AssertionError("kernel output did not match reference")

    with patch(
        "caiman_asr_train.tilelstm.autotune.tuners.tune", _tune_raises_assertion
    ):
        with pytest.raises(
            AssertionError, match="kernel output did not match reference"
        ):
            select_fastest.cache_clear()
            select_fwd_kernel(64, 64, device, torch.float16, torch.float32)

    select_fastest.cache_clear()


def test_select_fwd_kernel_skips_backend_when_tune_returns_none(identity_ranker):
    """
    ``tune`` returns None for an infeasible backend, so select_fastest skips it. When
    every backend is infeasible, select raises its own 'no candidate' AssertionError.
    """
    device = torch.device("cpu")  # no GPU needed -- tune and the ranker are stubbed

    def _tune_returns_none(*args, **kwargs):
        return None

    with patch("caiman_asr_train.tilelstm.autotune.tuners.tune", _tune_returns_none):
        with pytest.raises(AssertionError, match="no candidate kernel launched"):
            select_fastest.cache_clear()
            select_fwd_kernel(64, 64, device, torch.float16, torch.float32)

    select_fastest.cache_clear()


def test_select_fwd_kernel_reraises_unexpected_error(identity_ranker):
    """
    select_fastest has no try/except around ``tune``, so any (unexpected) error tune
    raises propagates rather than being silently swallowed.
    """
    device = torch.device("cpu")  # no GPU needed -- tune and the ranker are stubbed

    def _tune_raises_unknown(*args, **kwargs):
        raise RuntimeError("some unexpected internal bug")

    with patch("caiman_asr_train.tilelstm.autotune.tuners.tune", _tune_raises_unknown):
        with pytest.raises(RuntimeError, match="some unexpected internal bug"):
            select_fastest.cache_clear()
            select_fwd_kernel(64, 64, device, torch.float16, torch.float32)

    select_fastest.cache_clear()


@requires_tile_kernels
def test_select_fwd_kernel_is_cached(cap_autotune_configs):
    """
    Repeat calls at the same shape/dtypes return the identical cached kernel
    (no re-tune).
    """
    device = torch.device("cuda")
    k1 = select_fwd_kernel(64, 64, device, torch.float16, torch.float32)
    k2 = select_fwd_kernel(64, 64, device, torch.float16, torch.float32)
    assert k1 is k2


@requires_tile_kernels
def test_selection_disk_cache_roundtrip(tmp_path, monkeypatch, identity_ranker):
    """
    The first select runs a real par_compile sweep and writes per-config bench-cache
    entries; a fresh process at the same shape (simulated by clearing the in-process
    @cache) finds every config cached and does NOT re-sweep -- it only recompiles the
    winner.

    The ranker is stubbed (``identity_ranker``): this test redirects the cache root to
    an empty tmp_path, so the real ranker would either train a bundle from scratch in it
    or silently reuse a process-cached one, depending on test order.
    """
    import json
    from unittest import mock

    from caiman_asr_train.autotune import sweep
    from caiman_asr_train.tilelstm.autotune import tuners

    # Isolate the selection cache in tmp_path and ensure caching is enabled.
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TILELANG_DISABLE_CACHE", raising=False)
    monkeypatch.delenv("CAIMAN_AUTOTUNE_DISABLE_CACHE", raising=False)

    # This test isolates the cache in a fresh tmp_path, so the first tune is ALWAYS cold
    # (it can never hit a warm outer cache -- that is the point: it exercises the disk
    # WRITE path). The disk-cache roundtrip is grid-size-invariant, so cap each fwd
    # backend to a couple of feasible tiles to keep the cold sweep fast (~160s -> ~10s);
    # the full grid is covered by test_config_generators_yield_feasible_configs. Each
    # registry row's make_configs binds its generator eagerly (via _wrap_config), so cap
    # by swapping the _BACKENDS entry rather than the module-level generator name.
    import dataclasses

    from caiman_asr_train.tilelstm.autotune.kernel import Kernel

    def _cap(bk):
        capped = lambda *a, **k: bk.make_configs(*a, **k)[:2]  # noqa: E731
        return dataclasses.replace(bk, make_configs=capped)

    for k in (Kernel.fwd_c, Kernel.fwd_p):
        monkeypatch.setitem(tuners._BACKENDS, k, _cap(tuners._BACKENDS[k]))

    device = torch.device("cuda")
    H, B, S = 64, 64, 6

    # Spy on the GLOBAL par_compile (the miss-path delta bench). On a full cache hit
    # nothing is benched and only the winner is recompiled via factory.compile
    # (bypassing this alias), so the call count distinguishes a fresh sweep from a hit.
    with mock.patch.object(
        sweep, "_par_compile", side_effect=sweep._par_compile
    ) as spy:
        select_fastest.cache_clear()
        tuners.tune.cache_clear()
        select_fwd_kernel(H, B, device, torch.float16, torch.float32)
        swept = spy.call_count
        assert swept >= 1, "first tune should run a par_compile sweep"

        # Per-config bench-cache files now exist; each re-parses into a config + ms.
        files = list((tmp_path / "caiman_autotune_bench").rglob("*.json"))
        assert files, "expected per-config bench cache files"
        entry = json.loads(files[0].read_text())
        assert entry["config"]["H"] == H and "latency_ms" in entry

        # Fresh process: clear the in-memory memo, re-select -> disk hit, no sweep.
        select_fastest.cache_clear()
        tuners.tune.cache_clear()
        kernel = select_fwd_kernel(H, B, device, torch.float16, torch.float32)
        assert spy.call_count == swept, "second select must not re-sweep (disk hit)"

    # The disk-loaded kernel still runs the scan correctly.
    Y, C0, R, gates, Ys, Cs = scan_inputs(H, B, S, device)
    kernel(Y, C0, R, gates, Ys, Cs)
    ref_ys, ref_cs = lstm_ref(Y, C0, R, gates)
    assert_fwd_close(Ys, Cs, ref_ys, ref_cs)

    select_fastest.cache_clear()
    tuners.tune.cache_clear()
