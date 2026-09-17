"""``build_loss_fn`` must bake (dp_mu, dp_clamp) into every loss it can return."""

import pytest
import torch

from caiman_asr_train.rnnt import loss as loss_mod
from caiman_asr_train.rnnt.loss import LossSpec, build_loss_fn
from caiman_asr_train.tile_transducer.autotune import Mode, TileLangKernels
from caiman_asr_train.tile_transducer.utils import NO_CLAMP

BLANK = 1023

PENALTIES = [
    pytest.param(0.13, None, id="p4_recipe"),
    pytest.param(0.0, 0.02, id="clamped_only"),
    pytest.param(0.0, None, id="off"),
]


def spec_for(backend, dp_mu, dp_clamp):
    return LossSpec(
        backend=backend,
        blank_idx=BLANK,
        eos_idx=None,
        star_idx=None,
        packed_input=True,
        joint_n_hid=256,
        device=torch.device("cuda"),
        dtype=torch.float16,
        dp_mu=dp_mu,
        dp_clamp=dp_clamp,
    )


@pytest.fixture()
def stub_kernels(monkeypatch):
    """Selecting the real kernels would run the online autotuner."""

    def install(mode):
        kernels = TileLangKernels(
            mode=mode,
            fwd_run=lambda *a, **k: None,
            bwd_run=lambda *a, **k: None,
            dtype=torch.float16,
            accum_dtype=torch.float32,
        )
        monkeypatch.setattr(loss_mod, "_tilelang_kernels", lambda spec: kernels)

    return install


@pytest.mark.parametrize("dp_mu, dp_clamp", PENALTIES)
def test_triton_loss_is_built_with_the_penalty(dp_mu, dp_clamp):
    inner = build_loss_fn(spec_for("triton", dp_mu, dp_clamp)).t_loss
    assert (inner.dp_mu, inner.dp_clamp) == (dp_mu, dp_clamp)


@pytest.mark.parametrize("mode", [Mode.unfused, Mode.fused])
@pytest.mark.parametrize("dp_mu, dp_clamp", PENALTIES)
def test_tilelang_loss_is_built_with_the_penalty(stub_kernels, mode, dp_mu, dp_clamp):
    stub_kernels(mode)

    inner = build_loss_fn(spec_for("tilelang", dp_mu, dp_clamp)).t_loss

    assert inner.dp_mu == dp_mu
    assert inner.dp_gamma == (NO_CLAMP if dp_clamp is None else dp_clamp)


def test_cuda_backend_refuses_the_generalized_penalty():
    with pytest.raises(AssertionError, match="dp_mu"):
        build_loss_fn(spec_for("cuda", 0.13, None))

    with pytest.raises(AssertionError, match="dp_mu"):
        build_loss_fn(spec_for("cuda", 0.0, 0.02))

    build_loss_fn(spec_for("cuda", 0.0, None))
