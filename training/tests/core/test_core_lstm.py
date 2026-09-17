"""
Tests for the core ``LSTM`` backend dispatch (caiman_asr_train.core.lstm),
specifically the TileLang backend wiring.
"""

import subprocess
import sys

import pytest
import torch

from caiman_asr_train.autotune.support import tilelang_supported
from caiman_asr_train.core.lstm import LSTM, ConfigLSTM, TileConfig

requires_tile_kernels = pytest.mark.skipif(
    not tilelang_supported(),
    reason=(
        "TileLang LSTM kernels need a CUDA GPU of compute capability >= 8.0 "
        "(Ampere+) and torch >= 2.7; cp.async (T.async_copy) is unavailable below "
        "SM80, and the autotuner needs device properties older torch lacks"
    ),
)


def _config(backend):
    return ConfigLSTM(
        input_size=32,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        backend=backend,
        weights_init_scale=None,
        forget_gate_bias=None,
        hidden_hidden_bias_scale=None,
    )


def test_tile_backend_import_is_guarded():
    """
    Importing core.lstm (and building a non-Tile backend) must NOT pull in
    tilelang -- the TileLangLSTM import is guarded inside the ``TileConfig`` match
    case, exactly like the Custom/Legacy backends, so the heavy CUDA-only
    dependency stays off the startup / CPU-only path. Checked in a fresh
    interpreter so another test's import can't mask a regression.
    """
    code = (
        "import sys\n"
        "import caiman_asr_train.core.lstm as m\n"
        "m.LSTM(m.ConfigLSTM("
        "32, 64, 1, 0.0, m.TorchConfig(return_all=False), None, None, None))\n"
        "assert 'caiman_asr_train.tilelstm' not in sys.modules, 'tilelstm imported'\n"
        "assert 'tilelang' not in sys.modules, 'tilelang imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@requires_tile_kernels
def test_tile_backend_dispatches_and_runs(cap_autotune_configs):
    """
    A ``TileConfig`` LSTM builds a TileLangLSTM and returns the CustomLSTM 3-tuple
    (output, Hidden, AllHidden) with the right shapes.
    """
    from caiman_asr_train.tilelstm import TileLangLSTM

    model = LSTM(_config(TileConfig())).cuda()
    assert isinstance(model.lstm, TileLangLSTM)

    S, B, H = 5, 16, 64
    x = torch.randn(S, B, 32, device="cuda")
    with torch.autocast(device_type="cuda"):
        out, hidden, all_hidden = model(x)

    assert out.shape == (S, B, H)
    assert hidden.hidd.shape == (2, B, H) and hidden.cell.shape == (2, B, H)
    assert all_hidden is not None
    assert all_hidden.hidd.shape == (2, S, B, H)
    assert all_hidden.cell.shape == (2, S, B, H)
