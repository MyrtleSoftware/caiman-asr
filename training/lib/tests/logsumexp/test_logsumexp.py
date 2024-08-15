import numpy as np
import pytest
import torch
import torch.amp

from caiman_asr_train.utils.seed import set_seed

if not torch.cuda.is_available():
    pytest.skip("Cuda not available so can't run these test", allow_module_level=True)

import rnnt_ext.cuda.logsumexp as logsumexp_cu


def flatten(lval):
    return [item for sublist in lval for item in sublist]


def unique(lval):
    return list(set(lval))


def pows2(n):
    return [2**i for i in range(0, n)]


set_seed(42)


@pytest.mark.parametrize(
    "elems", unique(flatten([(x, x + 1, x + 3) for x in pows2(15)]))
)
@pytest.mark.parametrize("max_threads", pows2(10))
@pytest.mark.parametrize("promote", [True, False])
def test_logsumexp_full_precision(elems, max_threads, promote):
    x = torch.randn(1000, elems, device="cuda", dtype=torch.float64)

    # Compute logsumexp using torch
    t_lse = torch.logsumexp(x, dim=-1)

    # Compute logsumexp using our CUDA extension
    c_lse = logsumexp_cu.logsumexp(x, max_threads, promote)

    # Check that the results are the same
    assert torch.allclose(t_lse, c_lse)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize(
    "elems", unique(flatten([(x, x + 1, x + 3) for x in pows2(15)]))
)
@pytest.mark.parametrize("max_threads", [128])
@pytest.mark.parametrize("promote", [True])
def test_logsumexp(dtype, elems, max_threads, promote):
    #
    x_origin = torch.randn(1000, elems, device="cuda", dtype=torch.float64)

    ground_truth = torch.logsumexp(x_origin, dim=-1)

    # Reduced precision
    x = x_origin.to(dtype=dtype)

    with torch.amp.autocast("cuda"):
        # Compute logsumexp using torch
        t_lse = torch.logsumexp(x, dim=-1)

        c_lse = logsumexp_cu.logsumexp(x, max_threads, promote)

    t_err = (t_lse.to(torch.float64) - ground_truth).pow(2).sum().sqrt().item()
    c_err = (c_lse.to(torch.float64) - ground_truth).pow(2).sum().sqrt().item()

    print("torch err", t_err)
    print("cuda  err", c_err)

    # Check that the results are the same
    assert np.isclose(0, c_err) or (c_err <= 1.1 * t_err)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("elems", unique(flatten([(x + 1, x + 3) for x in pows2(15)])))
@pytest.mark.parametrize("max_threads", [128])
@pytest.mark.parametrize("promote", [True])
def test_log_softmax(dtype, elems, max_threads, promote):
    #
    x_origin = torch.randn(1000, elems, device="cuda", dtype=torch.float64)

    ground_truth = torch.log_softmax(x_origin, dim=-1)

    # Reduced precision
    x = x_origin.to(dtype=dtype)

    with torch.amp.autocast("cuda"):
        # Compute logsumexp using torch
        t_lse = torch.log_softmax(x, dim=-1)

    # Compute logsumexp using our CUDA extension
    sum = logsumexp_cu.logsumexp(x, max_threads, promote).unsqueeze(-1)

    assert sum.dtype == t_lse.dtype

    c_lse = x.to(sum.dtype) - sum

    t_err = (t_lse.to(torch.float64) - ground_truth).pow(2).sum().sqrt().item()
    c_err = (c_lse.to(torch.float64) - ground_truth).pow(2).sum().sqrt().item()

    print("torch err", t_err)
    print("cuda  err", c_err)

    # Check that the results are the same
    assert np.isclose(0, c_err) or (c_err <= 1.3 * t_err)
