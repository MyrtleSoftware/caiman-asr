import pytest

from caiman_asr_train.utils.math import ceil_div, round_down, round_up


def test_ceil_div():
    assert ceil_div(3, by=3) == 1
    assert ceil_div(-3, by=3) == -1

    assert ceil_div(10, by=3) == 4
    assert ceil_div(-10, by=3) == -3

    assert ceil_div(0, by=1) == 0
    assert ceil_div(-0, by=1) == 0

    with pytest.raises(ZeroDivisionError):
        ceil_div(1, by=0)

    with pytest.raises(ZeroDivisionError):
        ceil_div(0, by=0)


def test_round_up():
    assert round_up(3, multiple_of=3) == 3
    assert round_up(-3, multiple_of=3) == -3

    assert round_up(10, multiple_of=3) == 12
    assert round_up(-10, multiple_of=3) == -9

    assert round_up(0, multiple_of=1) == 0
    assert round_up(-0, multiple_of=1) == 0

    with pytest.raises(ValueError):
        round_up(1, multiple_of=0)

    with pytest.raises(ValueError):
        round_up(0, multiple_of=0)


def test_round_down():
    assert round_down(3, multiple_of=3) == 3
    assert round_down(-3, multiple_of=3) == -3

    assert round_down(10, multiple_of=3) == 9
    assert round_down(-10, multiple_of=3) == -12

    assert round_down(0, multiple_of=1) == 0
    assert round_down(-0, multiple_of=1) == 0

    with pytest.raises(ValueError):
        round_down(1, multiple_of=0)

    with pytest.raises(ValueError):
        round_down(0, multiple_of=0)
