#!/usr/bin/env python3
import pytest

from caiman_asr_train.train_utils.core import calculate_epoch


@pytest.mark.parametrize(
    "step, steps_per_epoch, epoch",
    [
        (1, 1, 1),
        (2, 1, 2),
        (3, 1, 3),
        (10, 10, 1),
        (11, 10, 2),
        (20, 10, 2),
        (21, 10, 3),
    ],
)
def test_calculate_epoch(step, steps_per_epoch, epoch):
    assert calculate_epoch(step, steps_per_epoch) == epoch
