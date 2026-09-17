"""Every loss backend against the reference, over shapes, lengths and layouts."""

import pytest
import torch
from tests.rnnt.loss.framework import (
    JOINT_BACKENDS,
    LOSS_BACKENDS,
    NO_TILELANG,
    assert_close,
    case,
    cells_case,
    joint_case,
    others,
    params,
    run,
    tilelang_blocks,
)

REF_LOSS, REF_JOINT = LOSS_BACKENDS[0], JOINT_BACKENDS[0]

CASES = dict(
    ragged=lambda: case(),
    single=lambda: case(B=1),
    uniform=lambda: case(uniform=True),
    y_len_zero=lambda: case(y_len_vals=[0, 0, 1, 5]),
    f_len_one=lambda: case(f_len_vals=[1, 7, 4, 7], y_len_vals=[3, 5, 0, 5]),
    long_tail=lambda: case(B=2, T=17, U=9),
)

JOINT_CASES = dict(
    ragged=lambda: joint_case(),
    single=lambda: joint_case(B=1),
    y_len_zero=lambda: joint_case(y_len_vals=[0, 0, 1, 31]),
    f_len_one=lambda: joint_case(f_len_vals=[1, 32, 8, 32], y_len_vals=[3, 5, 0, 31]),
    odd_shapes=lambda: joint_case(B=2, T=47, U=33, H=48, V=100),
    tall=lambda: joint_case(B=5, T=16, U=48),
)


def _compare(backend, ref, c, dtype=torch.float16, **kw):
    got = run(backend, c, dtype=dtype, **kw)
    want = run(ref, c, dtype=dtype, **kw)
    assert torch.isfinite(got.loss).all()
    assert_close(got.loss, want.loss, backend, dtype)
    for grad, grad_ref, mask in zip(got.grads, want.grads, c.grad_masks()):
        assert grad is not None
        assert_close(grad * mask, grad_ref * mask, backend, dtype)


@pytest.mark.parametrize("backend", params(others(LOSS_BACKENDS)))
@pytest.mark.parametrize("name", CASES)
def test_loss_matches_reference(backend, name):
    _compare(backend, REF_LOSS, CASES[name]())


@pytest.mark.parametrize("backend", params(others(JOINT_BACKENDS)))
@pytest.mark.parametrize("name", JOINT_CASES)
def test_joint_loss_matches_reference(backend, name):
    _compare(backend, REF_JOINT, JOINT_CASES[name]())


@NO_TILELANG
@pytest.mark.parametrize("block_C", [32, 128])
@pytest.mark.parametrize("kind", ["partial", "trailing_partial", "aligned"])
def test_tilelang_block_boundaries(block_C, kind):
    """A cell count below, over and on a multiple of the lattice block."""
    cells = dict(
        partial=max(25, block_C - 5),
        trailing_partial=2 * block_C + 5,
        aligned=3 * block_C,
    )[kind]
    _compare(tilelang_blocks(block_C), REF_LOSS, cells_case(cells))


@pytest.mark.parametrize("backend", [b for b in LOSS_BACKENDS if b.rect], ids=str)
def test_packed_matches_unpacked(backend):
    c = case()
    packed = run(backend, c, packed=True)
    rect = run(backend, c, packed=False)
    assert_close(packed.loss, rect.loss, backend, torch.float16)
    assert_close(packed.grads[0], c.pack(rect.grads[0]), backend, torch.float16)
