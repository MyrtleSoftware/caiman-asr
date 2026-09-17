"""The delay / EOS penalty: clamp(lam * ((T - 1) / 2 - t) / T**mu, -gamma, gamma)."""

import pytest
import torch
from tests.rnnt.loss.framework import (
    JOINT_BACKENDS,
    LOSS_BACKENDS,
    Mods,
    assert_close,
    case,
    joint_case,
    others,
    params,
    run,
    with_penalty,
)

from caiman_asr_train.tile_transducer.utils import NO_CLAMP, apply_emit_penalty

EOS = 126
EXACT = dict(rtol=0, atol=0)
PLAIN = [
    pytest.param(Mods(delay=0.3), id="delay"),
    pytest.param(Mods(delay=0.3, eos=0.4, eos_idx=EOS), id="delay_eos"),
]
GENERALIZED = [
    pytest.param(Mods(delay=0.3, eos=0.4, eos_idx=EOS, dp_mu=1.0), id="length_norm"),
    pytest.param(
        Mods(delay=0.3, eos=0.4, eos_idx=EOS, dp_mu=0.13, dp_clamp=0.02), id="p4_recipe"
    ),
    pytest.param(
        Mods(delay=0.3, eos=0.4, eos_idx=EOS, dp_mu=0.5, dp_clamp=0.3), id="sqrt_T"
    ),
]


def _compare(backend, ref, c, mods, dtype=torch.float16):
    got, want = run(backend, c, dtype, mods), run(ref, c, dtype, mods)
    assert_close(got.loss, want.loss, backend, dtype)
    for grad, grad_ref, mask in zip(got.grads, want.grads, c.grad_masks()):
        assert_close(grad * mask, grad_ref * mask, backend, dtype)


@pytest.mark.parametrize("backend", params(others(LOSS_BACKENDS)))
@pytest.mark.parametrize("mods", PLAIN)
def test_penalty_matches_reference(backend, mods):
    _compare(backend, LOSS_BACKENDS[0], case(eos_idx=mods.eos_idx), mods)


@pytest.mark.parametrize("backend", params(others(JOINT_BACKENDS)))
@pytest.mark.parametrize("mods", PLAIN)
def test_joint_penalty_matches_reference(backend, mods):
    _compare(backend, JOINT_BACKENDS[0], joint_case(eos_idx=mods.eos_idx), mods)


@pytest.mark.parametrize("backend", params(with_penalty(others(LOSS_BACKENDS))))
@pytest.mark.parametrize("mods", GENERALIZED)
def test_generalized_penalty_matches_reference(backend, mods):
    _compare(backend, LOSS_BACKENDS[0], case(eos_idx=EOS), mods)


@pytest.mark.parametrize("backend", params(with_penalty(others(JOINT_BACKENDS))))
@pytest.mark.parametrize("mods", GENERALIZED)
def test_generalized_joint_penalty_matches_reference(backend, mods):
    _compare(backend, JOINT_BACKENDS[0], joint_case(eos_idx=EOS), mods)


@pytest.mark.parametrize("backend", params(with_penalty(LOSS_BACKENDS)))
@pytest.mark.parametrize("dp_mu", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("lam", [0.5, 2.0])
def test_mu_norm_equals_scaled_unnormalized(backend, dp_mu, lam):
    """At one shared length T, T**mu normalization is just a rescaled lam."""
    c = case(uniform=True)
    T = c.max_f_len
    normalized = run(backend, c, torch.float32, Mods(delay=lam, dp_mu=dp_mu))
    scaled = run(backend, c, torch.float32, Mods(delay=lam / T**dp_mu))
    assert_close(normalized.loss, scaled.loss, backend, torch.float32)
    assert_close(normalized.grads[0], scaled.grads[0], backend, torch.float32)


@pytest.mark.parametrize("backend", params(with_penalty(LOSS_BACKENDS)))
def test_mu_norm_normalizes_per_item_length(backend):
    """mu=1 divides by each item's OWN frame count, not a batch-wide T."""
    lam = 2.0
    c = case(f_len_vals=[2, 4, 6, 7], y_len_vals=[3, 5, 0, 5])
    normalized = run(backend, c, torch.float32, Mods(delay=lam, dp_mu=1.0))
    for b, f_len in enumerate(c.f_len.tolist()):
        scaled = run(backend, c, torch.float32, Mods(delay=lam / f_len))
        assert_close(normalized.loss[b], scaled.loss[b], backend, torch.float32)


@pytest.mark.parametrize("backend", params(with_penalty(LOSS_BACKENDS)))
def test_loose_clamp_is_a_noop(backend):
    """A clamp above the peak bias must be exactly the unclamped penalty."""
    c = case(uniform=True)
    loose = run(backend, c, torch.float32, Mods(delay=0.5, dp_clamp=1e6))
    off = run(backend, c, torch.float32, Mods(delay=0.5))
    assert_close(loose.loss, off.loss, backend, **EXACT)
    assert_close(loose.grads[0], off.grads[0], backend, **EXACT)


@pytest.mark.parametrize("backend", params(with_penalty(LOSS_BACKENDS)))
def test_tight_clamp_approaches_no_penalty(backend):
    """A clamp at zero forces every per-edge bias to ~0."""
    c = case(uniform=True)
    clamped = run(backend, c, torch.float32, Mods(delay=2.0, dp_clamp=1e-9))
    none = run(backend, c, torch.float32)
    assert_close(clamped.loss, none.loss, backend, torch.float32, atol=1e-6)


def test_no_clamp_sentinel_agrees_with_triton():
    """``apply_emit_penalty`` defines its own sentinel rather than importing one."""
    from caiman_asr_train.rnnt.triton_loss import NO_CLAMP as TRITON_NO_CLAMP

    assert NO_CLAMP == TRITON_NO_CLAMP


@pytest.mark.parametrize(
    "dp_mu, dp_gamma",
    [(0.0, NO_CLAMP), (1.0, NO_CLAMP), (0.5, 0.35)],
    ids=["unnormalized", "length_norm", "sqrt_T_clamped"],
)
def test_apply_emit_penalty_matches_a_literal_loop(dp_mu, dp_gamma):
    torch.manual_seed(0)
    B, T, U = 3, 10, 6
    dev = "cuda"
    f_len = torch.tensor([10, 4, 7], device=dev, dtype=torch.int32)
    label = torch.randint(0, EOS, (B, U - 1), device=dev, dtype=torch.int32)
    label[0, 2] = EOS
    dp_lam, eos_lam = 0.2, 0.5

    def bias(lam, d, flen):
        return min(max(lam * d / flen**dp_mu, -dp_gamma), dp_gamma)

    lex = torch.randn(B, T, U, device=dev, dtype=torch.float32)
    want = lex.clone()
    for b in range(B):
        flen = int(f_len[b])
        for t in range(T):
            d = (flen - 1) / 2 - t
            for u in range(U):
                want[b, t, u] += bias(dp_lam, d, flen)
                if int(label[b, min(u, U - 2)]) == EOS:
                    want[b, t, u] += bias(eos_lam, d, flen)

    apply_emit_penalty(lex, f_len, label, dp_lam, eos_lam, EOS, dp_mu, dp_gamma)
    torch.testing.assert_close(lex, want, rtol=1e-5, atol=1e-5)


def test_apply_emit_penalty_zero_penalty_is_a_noop():
    dev = "cuda"
    lex = torch.randn(2, 4, 5, device=dev, dtype=torch.float32)
    before = lex.clone()
    f_len = torch.tensor([4, 2], device=dev, dtype=torch.int32)
    label = torch.randint(0, EOS, (2, 4), device=dev, dtype=torch.int32)

    apply_emit_penalty(lex, f_len, label, 0.0, 0.0, None)
    torch.testing.assert_close(lex, before, rtol=0, atol=0)
