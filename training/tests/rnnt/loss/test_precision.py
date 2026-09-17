"""Precision handling and poisoned logits."""

import pytest
import torch
from tests.rnnt.loss.framework import (
    CUDA,
    DTYPES,
    JOINT_BACKENDS,
    LOSS_BACKENDS,
    NO_TILELANG,
    NO_TRITON_BF16,
    TILELANG,
    TRITON,
    case,
    joint_case,
    module,
    others,
    params,
    run,
    with_fp64,
)

from caiman_asr_train.autotune.dtypes import accum_dtype_for
from caiman_asr_train.rnnt.triton_loss import TritonTransducerLoss

ACCUMULATING = [TRITON, TILELANG]


@pytest.mark.parametrize("backend,dtype", params(LOSS_BACKENDS, DTYPES()))
def test_gradient_keeps_the_input_precision(backend, dtype):
    got = run(backend, case(), dtype)
    assert got.grads[0].dtype is dtype
    assert torch.isfinite(got.grads[0]).all()


@pytest.mark.parametrize("backend,dtype", params(ACCUMULATING, DTYPES()))
def test_loss_accumulates_at_the_promoted_dtype(backend, dtype):
    assert run(backend, case(), dtype).loss.dtype is accum_dtype_for(dtype)


@pytest.mark.parametrize("backend", params(with_fp64(LOSS_BACKENDS)))
def test_fp32_is_fp32_accurate_and_fp16_is_not(backend):
    """The loss is computed at the incoming precision, not silently down-cast."""
    c = case()
    exact = run(backend, c, torch.float64).loss

    def rel(dtype):
        loss = run(backend, c, dtype).loss
        return ((loss.double() - exact).abs() / exact.abs()).max().item()

    fp32 = rel(torch.float32)
    assert fp32 < 1e-6, f"fp32 loss is not fp32-accurate: {fp32}"
    assert rel(torch.float16) > 20 * fp32, "fp16 loss unexpectedly as good as fp32"


@pytest.mark.parametrize(
    "amp_dtype",
    [torch.float16, pytest.param(torch.bfloat16, marks=NO_TRITON_BF16)],
)
def test_autocast_keeps_the_triton_accumulators_fp32(amp_dtype):
    c = case(uniform=True)
    x = c.leaf(torch.float32, packed=False)
    debug: list = []
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        loss = TritonTransducerLoss()(
            x, c.label, c.f_len, c.y_len, c.blank_idx, debug_list=debug
        )
    alpha, beta = debug
    assert (alpha.dtype, beta.dtype, loss.dtype) == ((torch.float32,) * 3)
    assert torch.isfinite(loss).all()

    ref = unpacked(TRITON, c)
    torch.testing.assert_close(
        loss,
        ref.loss,
        rtol=1e-2,
        atol={torch.float16: 1e-2, torch.bfloat16: 1e-1}[amp_dtype],
    )


def poisoned(index, value: float):
    c = case(uniform=True)
    c.logits[index] = value
    return c


def unpacked(backend, c):
    return run(backend, c, torch.float32, packed=False)


def test_neg_inf_blank_logit_stays_finite_and_matches_cuda():
    """A -inf blank logit blocks one lattice edge; other paths keep the loss finite."""
    c = poisoned((slice(None), 2, 0, 127), float("-inf"))
    got, want = unpacked(TRITON, c), unpacked(CUDA, c)
    assert torch.isfinite(got.loss).all()
    torch.testing.assert_close(got.loss, want.loss, rtol=1e-4, atol=1e-4)


def test_all_paths_blocked_gives_inf_loss():
    """log P = -inf, so the loss is +inf, not the NaN an unfixed logaddexp gives."""
    c = case(uniform=True)
    c.logits[0, 0, 0, c.blank_idx] = float("-inf")
    c.logits[0, 0, 0, int(c.label[0, 0])] = float("-inf")

    got = unpacked(TRITON, c)
    assert torch.isinf(got.loss[0]) and got.loss[0] > 0
    assert torch.isfinite(got.loss[1:]).all()
    assert not torch.isfinite(unpacked(CUDA, c).loss[0])


def test_pos_inf_logit_poisons_only_its_own_item():
    """+inf makes the denom +inf; the item must be poisoned, not laundered finite."""
    c = poisoned((0, 1, 1, 3), float("inf"))
    got, want = unpacked(TRITON, c), unpacked(CUDA, c)
    assert torch.isnan(got.loss[0]) and torch.isnan(want.loss[0])
    assert not torch.isfinite(got.grads[0][0]).all()
    assert torch.isfinite(got.loss[1:]).all()
    assert torch.isfinite(got.grads[0][1:]).all()
    torch.testing.assert_close(got.loss[1:], want.loss[1:], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        got.grads[0][1:], want.grads[0][1:], rtol=1e-4, atol=1e-4
    )


@NO_TILELANG
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_tilelang_gradient_aliases_the_input_logits(dtype):
    """dX is formed in place whatever the precision -- no copy."""
    c = case()
    x = c.leaf(dtype, packed=True)
    seen: list = []
    x.register_hook(lambda g: seen.append((g.data_ptr(), g.dtype)))
    module(TILELANG, c, dtype)(x).sum().backward()
    assert seen == [(x.data_ptr(), dtype)]


@NO_TILELANG
def test_tilelang_falls_back_to_a_copy_when_not_contiguous():
    """Non-contiguity is the only remaining reason to stage a copy."""
    c = case()
    wide = torch.zeros(c.total, 2 * c.V, device="cuda", dtype=torch.float16)
    wide[:, : c.V] = c.pack(c.logits)
    x = wide.detach()[:, : c.V].requires_grad_(True)
    assert not x.is_contiguous()
    before = x.detach().clone()

    loss = module(TILELANG, c, torch.float16)(x)
    loss.backward(torch.ones_like(loss))

    assert x.grad.dtype == x.dtype and x.grad.shape == x.shape
    assert x.grad.data_ptr() != x.data_ptr()
    torch.testing.assert_close(x.detach(), before, rtol=0, atol=0)


@pytest.mark.parametrize("backend", params(others(JOINT_BACKENDS)))
@pytest.mark.parametrize("joint_dtype", [torch.float16, torch.float32])
def test_fp32_leaves_keep_their_grads_in_fp32(backend, joint_dtype):
    """The training path hands fp32 leaves in, so the cast happens inside the graph."""
    c = joint_case()
    got = run(backend, c, torch.float32, joint_dtype=joint_dtype)
    assert got.loss.dtype is torch.float32
    for grad in got.grads:
        assert grad.dtype is torch.float32
