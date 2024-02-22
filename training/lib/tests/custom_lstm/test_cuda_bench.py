from math import sqrt
from statistics import mean, stdev

import pytest
import torch

if not torch.cuda.is_available():
    # Before jit import to avoid jit compilation on when it won't be used.
    pytest.skip("Cuda not available so can't run these test", allow_module_level=True)

from rnnt_ext.custom_lstm.lstm import CustomLSTM

scaler = torch.cuda.amp.GradScaler()

cross = torch.nn.CrossEntropyLoss()


def time(rnn, X, h0, c0, target, amp, runs):
    def forward():
        if amp:
            with torch.autocast(device_type="cuda"):
                output, *_ = rnn(X, (h0, c0))
                loss = cross(output, target)
                return loss
        else:
            output, *_ = rnn(X, (h0, c0))
            loss = cross(output, target)
            return loss

    def backward(loss):
        if amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    def event():
        return torch.cuda.Event(enable_timing=True)

    events = [(event(), event()) for _ in range(runs)]

    torch.cuda.synchronize()

    for a, b in events:
        a.record()
        backward(forward())
        b.record()

    torch.cuda.synchronize()

    return [a.elapsed_time(b) for a, b in events]


def bench(rnn, X, h, c, target, amp):
    # Warmup
    time(rnn, X, h, c, target, amp, 3)

    times = []

    def frac_err(x):
        return stdev(x) / mean(x) / sqrt(len(x))

    tot = 0  # milliseconds

    while True:
        new = time(rnn, X, h, c, target, amp, 3)

        times.extend(new)
        tot += sum(new)

        if tot > 100 or frac_err(times) < 0.10:
            break
    # Return the mean and standard error
    # https://en.wikipedia.org/wiki/Standard_error
    return mean(times), stdev(times) / sqrt(len(times))


def comp(ref, test, frac):
    if ref[0] > test[0]:
        # Being faster
        return True
    elif ref[0] + ref[1] >= test[0] - test[1]:
        # Being within error
        return True
    else:
        # Being slower
        return (test[0] - ref[0]) / ref[0] < frac


@pytest.mark.parametrize("batch", [1, 8, 64])
@pytest.mark.parametrize("hidden", [256, 512])
@pytest.mark.parametrize("seq", [256, 512])
@pytest.mark.parametrize("amp", [True, False])
@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cuda_bench(batch, hidden, seq, amp, layers, dtype):
    # NOTE: If hidden gets too small then cuDNN will switch over to a persistent LSTM
    # which this code cannot compete with.

    if amp and dtype is torch.float64:
        # AMP is a noop under double precision.
        return

    input = hidden
    device = torch.device("cuda")

    kwargs = {"dtype": dtype, "device": device}

    X = torch.randn(
        seq,
        batch,
        input,
        **kwargs,
        requires_grad=True,
    )

    h0 = torch.randn(layers, batch, hidden, **kwargs)
    c0 = torch.randn(layers, batch, hidden, **kwargs)

    target = torch.randn_like(X)

    # Benchmarking

    ref = bench(torch.nn.LSTM(input, hidden, layers, **kwargs), X, h0, c0, target, amp)

    soft = bench(
        CustomLSTM(input, hidden, layers, hard=False, **kwargs), X, h0, c0, target, amp
    )

    hard = bench(
        CustomLSTM(input, hidden, layers, hard=True, **kwargs), X, h0, c0, target, amp
    )

    assert comp(ref, soft, 0.2), f"Soft={soft}, Ref={ref} - (mean, stderr)"
    assert comp(ref, hard, 0.2), f"Hard={hard}, Ref={ref} - (mean, stderr)"
