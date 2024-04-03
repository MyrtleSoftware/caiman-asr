#!/usr/bin/env python3
import pytest

from caiman_asr_train.train_utils.rsp import generate_batch_history


@pytest.mark.parametrize("a", [10, 5])
@pytest.mark.parametrize("b", [2])
def test_generate_batch_history(a, b):
    trials = 10**4
    lyst = [generate_batch_history([a, b]) for _ in range(trials)]
    assert lyst.count(1) + lyst.count(2) == trials
    assert lyst.count(1) / trials == pytest.approx(a / (a + b), abs=0.01)
    lyst2 = [generate_batch_history([a, 0, b]) for _ in range(trials)]
    assert lyst2.count(1) + lyst2.count(3) == trials
    assert lyst2.count(1) / trials == pytest.approx(a / (a + b), abs=0.01)
    # Just to make sure it's random
    assert lyst2.count(1) != lyst.count(1)
