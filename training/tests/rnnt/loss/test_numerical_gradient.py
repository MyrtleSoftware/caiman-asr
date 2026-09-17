"""Numerical gradient checks, the only non-comparative tests in this suite.

Neither TileLang backend forms dX in place so torch.autograd.gradcheck cannot be used.
Their correctness is verified by comparing to the reference backend in ``test_equivalence``.
"""

import pytest
import torch
from tests.rnnt.loss.framework import (
    LAYOUTS,
    LOSS_BACKENDS,
    Mods,
    case,
    loss_only,
    params,
    with_fp64,
    with_gradcheck,
    with_penalty,
)

SMALL = dict(B=1, T=3, U=3, V=32)
GRADCHECK = dict(eps=1e-6, atol=1e-5, rtol=1e-3)

PENALTIES = [
    pytest.param(Mods(delay=2.0), id="delay"),
    pytest.param(Mods(delay=2.0, eos=0.5, eos_idx=30), id="delay_eos"),
    pytest.param(Mods(delay=2.0, dp_mu=0.5, dp_clamp=1.0), id="clamped_sqrt_T"),
    pytest.param(Mods(delay=2.0, dp_mu=1.0), id="length_norm"),
]

GRADCHECKED = with_gradcheck(with_fp64(LOSS_BACKENDS))


@pytest.mark.parametrize("backend,packed", params(GRADCHECKED, LAYOUTS))
def test_backward_matches_numerical_gradient(backend, packed):
    c = case(**SMALL)
    x = c.leaf(torch.float64, packed)
    assert torch.autograd.gradcheck(
        loss_only(backend, c, torch.float64, packed=packed), [x], **GRADCHECK
    )


@pytest.mark.parametrize("backend,packed", params(with_penalty(GRADCHECKED), LAYOUTS))
@pytest.mark.parametrize("mods", PENALTIES)
def test_penalty_backward_matches_numerical_gradient(backend, packed, mods):
    c = case(**SMALL, eos_idx=mods.eos_idx)
    x = c.leaf(torch.float64, packed)
    assert torch.autograd.gradcheck(
        loss_only(backend, c, torch.float64, mods=mods, packed=packed), [x], **GRADCHECK
    )
