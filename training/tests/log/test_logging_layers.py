#!/usr/bin/env python3
from unittest.mock import MagicMock

import torch
from torch.amp import GradScaler

from caiman_asr_train.log.logging_layers import get_logging_entries


def tiny_model_with_grads():
    model = torch.nn.Linear(4, 2)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    return model


def scaler_mock(scale: float) -> MagicMock:
    scaler = MagicMock(spec=GradScaler)
    scaler.get_scale.return_value = scale
    return scaler


def test_get_logging_entries_no_scaler():
    total_norm, per_layer, log2_scaler = get_logging_entries(
        tiny_model_with_grads(), None
    )
    assert log2_scaler == 0.0
    assert total_norm > 0.0
    assert per_layer


def test_get_logging_entries_normal_scale():
    _, _, log2_scaler = get_logging_entries(
        tiny_model_with_grads(), scaler_mock(2.0**16)
    )
    assert log2_scaler == 16.0


def test_get_logging_entries_zero_scale_does_not_raise():
    """A GradScaler that backs off far enough underflows its scale to 0.0.
    Logging must survive that (it used to die with a math domain error)."""
    total_norm, _, log2_scaler = get_logging_entries(
        tiny_model_with_grads(), scaler_mock(0.0)
    )
    assert log2_scaler == float("-inf")
    assert total_norm == float("inf")  # grads unscale to inf; norm reflects that
