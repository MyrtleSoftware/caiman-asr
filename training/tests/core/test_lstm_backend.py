import pytest
import torch

from caiman_asr_train.core.lstm import (
    LSTM,
    ConfigLSTM,
    CustomConfig,
    Hidden,
    TorchConfig,
)

if not torch.cuda.is_available():
    # The custom backend requires CUDA, so we can't compare against it on CPU.
    pytest.skip(
        "Custom LSTM requires CUDA so can't compare backends",
        allow_module_level=True,
    )


def make_lstm(backend, input_size, hidden_size, num_layers):
    return LSTM(
        ConfigLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.0,
            backend=backend,
            weights_init_scale=None,
            forget_gate_bias=None,
            hidden_hidden_bias_scale=None,
        )
    )


def copy_params(dst, src):
    """
    Copy LSTM parameters src -> dst.

    Both torch.nn.LSTM and CustomLSTM expose num_layers and nn.LSTM-style
    per-layer weight/bias attributes (see test_cuda.py::for_param_pairs).
    """
    assert dst.num_layers == src.num_layers
    for i in range(dst.num_layers):
        for pattern in ["weight_ih_l{}", "weight_hh_l{}", "bias_ih_l{}", "bias_hh_l{}"]:
            getattr(dst, pattern.format(i)).data.copy_(
                getattr(src, pattern.format(i)).data
            )


@pytest.mark.parametrize("num_layers", [1, 2, 3])
@pytest.mark.parametrize("seq_length", [1, 5, 8])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("use_initial_state", [False, True])
def test_torch_return_all_matches_custom(
    num_layers, seq_length, batch_size, use_initial_state
):
    """
    The torch backend with return_all=True must reproduce the custom
    backend's AllHidden (and output / final Hidden) given identical weights.
    """
    torch.manual_seed(42)

    input_size, hidden_size = 16, 16

    torch_lstm = (
        make_lstm(TorchConfig(return_all=True), input_size, hidden_size, num_layers)
        .cuda()
        .eval()
    )
    custom_lstm = (
        make_lstm(CustomConfig(0.0), input_size, hidden_size, num_layers).cuda().eval()
    )

    # Custom backend is the reference; copy its weights into the torch backend.
    with torch.no_grad():
        copy_params(torch_lstm.lstm, custom_lstm.lstm)

    x = torch.randn(seq_length, batch_size, input_size, device="cuda")

    if use_initial_state:
        h0 = Hidden(
            hidd=torch.randn(num_layers, batch_size, hidden_size, device="cuda"),
            cell=torch.randn(num_layers, batch_size, hidden_size, device="cuda"),
        )
    else:
        h0 = None

    with torch.no_grad():
        x_torch, h_torch, all_torch = torch_lstm(x, h0)
        x_custom, h_custom, all_custom = custom_lstm(x, h0)

    assert all_torch is not None, "return_all=True should populate AllHidden"
    assert all_custom is not None, "custom backend always returns AllHidden"

    # Stepping cuDNN one timestep at a time diverges from the custom kernel's
    # full-sequence fp32 maths by ~1e-4 (grows with layers/timesteps), so use a
    # looser bound than test_cuda.py's full-sequence 1e-6 comparison.
    tol = 1e-3

    def close(a, b, name):
        assert torch.allclose(a, b, atol=tol), f"{name} max err = {(a - b).abs().max()}"

    # The per-timestep states are what return_all exists to provide.
    close(all_torch.hidd, all_custom.hidd, "all_h.hidd")
    close(all_torch.cell, all_custom.cell, "all_h.cell")
    # Output sequence and final hidden/cell state must also still match.
    close(x_torch, x_custom, "output")
    close(h_torch.hidd, h_custom.hidd, "final hidd")
    close(h_torch.cell, h_custom.cell, "final cell")
