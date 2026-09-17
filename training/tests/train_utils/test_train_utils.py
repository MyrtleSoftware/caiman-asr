#!/usr/bin/env python3
import pytest
import torch

from caiman_asr_train.train_utils.core import calculate_epoch, is_loss_invalid


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


@pytest.mark.parametrize(
    "values, invalid",
    [
        ([1.0], False),
        ([-5.0], False),  # negative loss is legal (delay penalty)
        ([float("nan")], True),
        ([float("inf")], True),  # inf must be dropped too, not only NaN
        ([float("-inf")], True),
        ([1.0, float("inf")], True),
        ([1.0, 2.0], False),
    ],
)
def test_is_loss_invalid(values, invalid):
    assert is_loss_invalid(torch.tensor(values), num_gpus=1) is invalid
