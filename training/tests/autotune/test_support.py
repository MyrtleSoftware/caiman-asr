"""The compute-capability gate that picks between the TileLang/Triton kernels and the
custom CUDA fallbacks."""

import pytest
import torch

from caiman_asr_train.autotune.support import sm80_supported


@pytest.mark.parametrize(
    "capability, cuda_available, expected",
    [
        ((8, 0), True, True),  # A100 (sm_80): the inclusive boundary
        ((8, 6), True, True),  # Ampere sm_86 (RTX 3090 / A6000)
        ((8, 9), True, True),  # Ada sm_89 (RTX 4090)
        ((9, 0), True, True),  # Hopper (sm_90)
        ((12, 0), True, True),  # Blackwell
        ((7, 5), True, False),  # Turing sm_75 (T4): below SM80, no cp.async
        ((7, 0), True, False),  # Volta sm_70 (V100)
        ((9, 0), False, False),  # no CUDA device
    ],
)
def test_sm80_supported(monkeypatch, capability, cuda_available, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    assert sm80_supported() is expected
