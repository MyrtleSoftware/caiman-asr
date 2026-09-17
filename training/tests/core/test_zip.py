import builtins

import pytest
import torch
from beartype import beartype
from beartype.roar import BeartypeCallHintViolation

from caiman_asr_train.core.lstm import Hidden
from caiman_asr_train.core.model import StateEncoder, StatePred
from caiman_asr_train.core.zip import _unzip, _zip, unzip, zip

L, H = 2, 5


@beartype
def rand_hidden(B: int) -> Hidden:
    return Hidden(hidd=torch.randn(L, B, H), cell=torch.randn(L, B, H))


@beartype
def rand_state_encoder(B: int) -> StateEncoder:
    return StateEncoder(pre_stack=rand_hidden(B), aft_stack=rand_hidden(B))


@beartype
def rand_state_pred(B: int) -> StatePred:
    return StatePred(state=rand_hidden(B), embed=torch.randn(1, B, H))


@beartype
def assert_hidden_equal(a: Hidden, b: Hidden) -> None:
    assert torch.equal(a.hidd, b.hidd)
    assert torch.equal(a.cell, b.cell)


@beartype
def assert_state_equal(
    a: StateEncoder | StatePred, b: StateEncoder | StatePred
) -> None:
    match a, b:
        case StateEncoder(), StateEncoder():
            assert_hidden_equal(a.pre_stack, b.pre_stack)
            assert_hidden_equal(a.aft_stack, b.aft_stack)
        case StatePred(), StatePred():
            assert_hidden_equal(a.state, b.state)
            assert torch.equal(a.embed, b.embed)
        case _:
            assert False, f"Type mismatch: {type(a)} vs {type(b)}"


def test_zip_hidden_concatenates_in_order():
    xs = [rand_hidden(1) for _ in range(4)]

    z = _zip(xs)

    assert z.hidd.shape == (L, 4, H)
    assert z.cell.shape == (L, 4, H)
    for i, x in enumerate(xs):
        assert torch.equal(z.hidd[:, i], x.hidd[:, 0])
        assert torch.equal(z.cell[:, i], x.cell[:, 0])


def test_unzip_hidden_slices_batch():
    x = rand_hidden(3)

    xs = _unzip(x)

    assert len(xs) == 3
    for i, xi in enumerate(xs):
        assert xi.hidd.shape == (L, 1, H)
        assert torch.equal(xi.hidd[:, 0], x.hidd[:, i])
        assert torch.equal(xi.cell[:, 0], x.cell[:, i])


def test_zip_unzip_hidden_roundtrip():
    xs = [rand_hidden(1) for _ in range(3)]
    for a, b in builtins.zip(xs, _unzip(_zip(xs)), strict=True):
        assert_hidden_equal(a, b)

    x = rand_hidden(4)
    assert_hidden_equal(x, _zip(_unzip(x)))


def test_zip_hidden_rejects_bad_input():
    with pytest.raises(AssertionError, match="at least one"):
        _zip([])

    with pytest.raises(AssertionError, match="batch size 1"):
        _zip([rand_hidden(2)])

    with pytest.raises(AssertionError, match="same shape"):
        _zip(
            [
                rand_hidden(1),
                Hidden(hidd=torch.randn(L, 1, H + 1), cell=torch.randn(L, 1, H + 1)),
            ]
        )


@pytest.mark.parametrize("make", [rand_state_encoder, rand_state_pred])
def test_zip_unzip_state_roundtrip(make):
    states = [make(1) for _ in range(3)]

    z = zip(states)
    assert isinstance(z, type(states[0]))

    for a, b in builtins.zip(states, unzip(z), strict=True):
        assert_state_equal(a, b)

    batched = make(4)
    parts = unzip(batched)
    assert len(parts) == 4
    assert_state_equal(batched, zip(parts))


def test_zip_state_encoder_values():
    states = [rand_state_encoder(1) for _ in range(3)]

    z = zip(states)

    assert_hidden_equal(z.pre_stack, _zip([s.pre_stack for s in states]))
    assert_hidden_equal(z.aft_stack, _zip([s.aft_stack for s in states]))


def test_zip_state_pred_values():
    states = [rand_state_pred(1) for _ in range(3)]

    z = zip(states)

    assert z.embed.shape == (1, 3, H)
    assert_hidden_equal(z.state, _zip([s.state for s in states]))
    for i, s in enumerate(states):
        assert torch.equal(z.embed[:, i], s.embed[:, 0])


def test_zip_state_rejects_bad_input():
    with pytest.raises(AssertionError, match="at least one"):
        zip([])

    # Mixed lists are rejected either by beartype's container check or, if the
    # sampled element happens to type-check, by an AttributeError when the
    # wrong-typed element is unpacked.
    with pytest.raises((AttributeError, BeartypeCallHintViolation)):
        zip([rand_state_encoder(1), rand_state_pred(1)])

    with pytest.raises(AssertionError, match="batch size 1"):
        zip([rand_state_pred(2)])

    with pytest.raises(AssertionError, match="same shape"):
        zip(
            [
                StatePred(state=rand_hidden(1), embed=torch.randn(1, 1, H)),
                StatePred(state=rand_hidden(1), embed=torch.randn(1, 1, H + 1)),
            ]
        )
