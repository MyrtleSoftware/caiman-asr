"""End-to-end kernel selection: tune all four kernels, then pick the cheaper mode from
the stage latencies the tuner measured. ``cap_configs`` truncates each candidate list
(as ``tests/conftest.py::cap_autotune_configs`` does for the LSTM) to keep the compile
budget to a handful of kernels."""

import dataclasses

import pytest
import torch
from tests.sm80 import skip_without_tilelang

skip_without_tilelang()

from caiman_asr_train.tile_transducer.autotune import tuners  # noqa: E402
from caiman_asr_train.tile_transducer.autotune import (  # noqa: E402
    Mode,
)
from caiman_asr_train.tile_transducer.autotune import select as select_mod  # noqa: E402
from caiman_asr_train.tile_transducer.autotune import (  # noqa: E402
    select_tilelang_kernels,
)
from caiman_asr_train.tile_transducer.autotune.kernel import Kernel  # noqa: E402
from caiman_asr_train.tile_transducer.autotune.mock_batch import (  # noqa: E402
    TuneShape,
)

MAX_TEST_CONFIGS = 3
H, V = 64, 256
BLANK = V - 1
TEST_SHAPE = TuneShape(cells=1024)


@pytest.fixture()
def cap_configs(monkeypatch, tmp_path):
    def cap(bk):
        def capped(*a, **k):
            return bk.make_configs(*a, **k)[:MAX_TEST_CONFIGS]

        return dataclasses.replace(bk, make_configs=capped)

    for k in list(tuners._BACKENDS):
        monkeypatch.setitem(tuners._BACKENDS, k, cap(tuners._BACKENDS[k]))
    monkeypatch.setattr(select_mod, "tune_shape", lambda: TEST_SHAPE)
    tuners.tune.cache_clear()
    select_tilelang_kernels.cache_clear()
    yield
    tuners.tune.cache_clear()
    select_tilelang_kernels.cache_clear()


def test_every_kernel_has_feasible_configs():
    """An empty candidate list aborts the tune at the start of a train."""
    device = torch.device("cuda")
    for kern in Kernel:
        configs = tuners._BACKENDS[kern].make_configs(H, V, BLANK, device)
        assert configs, f"no feasible config for {kern}"
        for cfg in configs:
            assert cfg["threads"] % 32 == 0


def test_select_picks_a_mode_and_returns_working_runners(cap_configs):
    choice = select_tilelang_kernels(H, V, BLANK, torch.device("cuda"))

    assert choice.mode in tuple(Mode)
    assert callable(choice.fwd_run) and callable(choice.bwd_run)
    assert (choice.dtype, choice.accum_dtype) == (torch.float16, torch.float32)


def test_fp64_is_refused():
    """The fused backward gemms a register fragment, which TileLang cannot do in fp64."""
    with pytest.raises(AssertionError, match="fp64"):
        select_tilelang_kernels(H, V, BLANK, torch.device("cuda"), torch.float64)


def test_mode_follows_the_summed_stage_latencies(monkeypatch):
    latencies = {
        Kernel.proj_lse: 10.0,
        Kernel.proj_and_loss_bwd: 10.0,
        Kernel.logsumexp: 4.0,
        Kernel.transducer_loss_bwd: 4.0,
    }

    def fake_tune(kernel, *a, **k):
        return None, [
            {"config": {"kernel": str(kernel)}, "latency_ms": latencies[kernel]}
        ]

    monkeypatch.setattr(select_mod, "tune", fake_tune)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    args = (H, V, BLANK, torch.device("cuda"), TEST_SHAPE, torch.float16, torch.float32)

    assert select_mod._select_mode(*args)["mode"] == Mode.unfused.value

    latencies[Kernel.logsumexp] = 40.0
    assert select_mod._select_mode(*args)["mode"] == Mode.fused.value
