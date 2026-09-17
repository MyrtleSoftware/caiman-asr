"""
Tests for the ``TileLangLSTM`` nn.Module wiring -- construction, parameter
naming, and checkpoint interop with ``CustomLSTM`` (the module it is a drop-in
for). Distinct from the recurrent-scan kernel tests in ``test_lstm_fwd`` /
``test_lstm_bwd``.

These are CPU-only: they construct the module and inspect its ``state_dict``; no
kernel is built or launched.
"""

import pytest
import torch

from caiman_asr_train.tilelstm import TileLangLSTM


def _expected_flat_keys(num_layers: int) -> set[str]:
    """The flat PyTorch-LSTM parameter names CustomLSTM/nn.LSTM expose."""
    return {
        f"{w}_l{i}"
        for i in range(num_layers)
        for w in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
    }


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_state_dict_is_flat_pytorch_lstm_naming(num_layers):
    """
    TileLangLSTM's state_dict must be exactly the flat PyTorch-LSTM parameter
    names (``weight_ih_l{i}`` ...) and NOTHING else.

    Regression guard for the nn.ModuleList duplicate-key bug: registering the
    layers in an nn.ModuleList AND under flat aliases emitted both
    ``layers.{i}.weight_ih`` and ``weight_ih_l{i}`` -- doubling every weight and
    breaking strict checkpoint interop with CustomLSTM / nn.LSTM.
    """
    tl = TileLangLSTM(8, 16, num_layers=num_layers)
    keys = set(tl.state_dict().keys())
    assert keys == _expected_flat_keys(num_layers)
    assert not any(k.startswith("layers.") for k in keys)


@pytest.mark.parametrize("num_layers", [1, 2])
def test_state_dict_matches_custom_lstm(num_layers):
    """
    TileLangLSTM is a drop-in for CustomLSTM, so their state_dicts must carry the
    SAME keys and a checkpoint from one must load into the other under
    ``strict=True`` -- the actual swap scenario the flat parameter names exist for.
    """
    from rnnt_ext.custom_lstm.lstm import CustomLSTM

    tl = TileLangLSTM(8, 16, num_layers=num_layers)
    custom = CustomLSTM(8, 16, num_layers=num_layers)

    assert set(tl.state_dict().keys()) == set(custom.state_dict().keys())

    # Both load directions succeed and actually copy the weights.
    tl.load_state_dict(custom.state_dict(), strict=True)
    custom.load_state_dict(tl.state_dict(), strict=True)
    for i in range(num_layers):
        torch.testing.assert_close(
            getattr(tl, f"weight_ih_l{i}"), getattr(custom, f"weight_ih_l{i}")
        )
