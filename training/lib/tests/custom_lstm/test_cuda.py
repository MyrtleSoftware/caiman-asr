from random import choice, sample

import pytest
import torch

if not torch.cuda.is_available():
    # Before jit import to avoid jit compilation on when we wont use it.
    pytest.skip("Cuda not available so can't run these test", allow_module_level=True)

import rnnt_ext.custom_lstm.lstm as CUDA
from rnnt_ext.custom_lstm.legacy import CustomLSTM as Legacy


@pytest.mark.parametrize("seq_length", [1, 7])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("input_size", [1, 2])
@pytest.mark.parametrize("hidden_size", [1, 5])
@pytest.mark.parametrize("layer_function", [CUDA.soft_layer_fun, CUDA.soft_layer_fun])
def test_derivatives(seq_length, batch_size, input_size, hidden_size, layer_function):
    #
    kwargs = {
        "dtype": torch.float64,
        "device": torch.device("cuda"),
        "requires_grad": True,
    }

    X = torch.randn(seq_length, batch_size, input_size, **kwargs)

    h0 = torch.randn(batch_size, hidden_size, **kwargs)
    c0 = torch.randn(batch_size, hidden_size, **kwargs)

    R = torch.randn(4 * hidden_size, hidden_size, **kwargs)
    W = torch.randn(4 * hidden_size, input_size, **kwargs)

    BW = torch.randn(4 * hidden_size, **kwargs)
    BR = torch.randn(4 * hidden_size, **kwargs)

    variables = [X, W, R, BW, BR]

    def stub(X, W, R, BW, BR):
        out, _ = layer_function.apply(h0, c0, X, W, R, BW, BR)
        return out

    assert torch.autograd.gradcheck(stub, variables)


def for_param_pairs(a, b, fn):
    assert a.num_layers == b.num_layers, "Number of layers in each LSTM must match"

    for i in range(a.num_layers):
        for pattern in ["weight_ih_l{}", "weight_hh_l{}", "bias_ih_l{}", "bias_hh_l{}"]:
            fn(getattr(a, pattern.format(i)), getattr(b, pattern.format(i)))


def check_values(
    candidate,
    ref,
    num_layers,
    seq_length,
    batch_size,
    input_size,
    hidden_size,
    dtype,
    amp,
    tol,
):
    """
    Expects candidate and ref to lstm's with the same internal parameters
    """

    X1 = torch.randn(
        seq_length,
        batch_size,
        input_size,
        requires_grad=True,
        dtype=dtype,
        device=torch.device("cuda"),
    )

    X2 = X1.detach().clone()

    X2.requires_grad = True

    h0 = torch.randn(
        num_layers,
        batch_size,
        hidden_size,
        dtype=dtype,
        device=torch.device("cuda"),
    )

    c0 = torch.randn_like(h0)

    cross = torch.nn.CrossEntropyLoss()

    def forward(rnn, X):
        if amp:
            with torch.autocast(device_type="cuda"):
                output, (hn, cn) = rnn(X, (h0, c0))

                # One hot target
                target = torch.zeros_like(output)

                target[0, 0] = 1

                loss = cross(output, target)

                return output, (hn, cn), loss
        else:
            output, (hn, cn) = rnn(X, (h0, c0))

            # One hot target
            target = torch.zeros_like(output)

            target[0, 0] = 1

            loss = cross(output, target)

            return output, (hn, cn), loss

    o1, _, l1 = forward(ref, X1)
    o2, _, l2 = forward(candidate, X2)

    assert torch.allclose(o1, o2, atol=tol), f"max err = {(o1 - o2).max()}"

    l1.backward()
    l2.backward()

    def check_grad(a, b):
        assert torch.allclose(
            a.grad, b.grad, atol=tol
        ), f"max err = {(a.grad - b.grad).max()}"

    check_grad(X1, X2)

    for_param_pairs(candidate, ref, check_grad)


def build_lstm_pair(num_layers, input_size, hidden_size, hard, dtype, use_legacy=False):
    #

    candidate = CUDA.CustomLSTM(
        input_size,
        hidden_size,
        num_layers,
        hard=hard,
        dtype=dtype,
        device=torch.device("cuda"),
    )

    if use_legacy or hard:
        ref = Legacy(
            input_size,
            hidden_size,
            num_layers,
            hard=hard,
        ).to(device=torch.device("cuda"), dtype=dtype)

    else:
        ref = torch.nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dtype=dtype,
            device=torch.device("cuda"),
        )

    with torch.no_grad():
        for_param_pairs(
            candidate, ref, lambda can_param, ref_param: ref_param.copy_(can_param)
        )

    return candidate, ref


@pytest.mark.parametrize("seq_length", [1, 8])
@pytest.mark.parametrize("num_layers", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("input_size", [1, 2, 16])
@pytest.mark.parametrize("hidden_size", [1, 16])
@pytest.mark.parametrize("amp", [True, False])
@pytest.mark.parametrize("hard", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_values(
    seq_length, num_layers, batch_size, input_size, hidden_size, hard, amp, dtype
):
    if amp and dtype is torch.float64:
        # AMP is a noop under double precision.
        return

    if amp and hard:
        # Legacy does not support AMP
        return

    if amp:
        tol = 1e-03
    elif dtype is torch.float32:
        tol = 1e-06
    elif dtype is torch.float64:
        tol = 1e-12
    else:
        assert False, f"Unknown dtype {dtype}"

    candidate, ref = build_lstm_pair(
        num_layers, input_size, hidden_size, hard=hard, dtype=dtype
    )

    check_values(
        candidate,
        ref,
        num_layers,
        seq_length,
        batch_size,
        input_size,
        hidden_size,
        dtype,
        amp=amp,
        tol=tol,
    )
