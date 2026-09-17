"""
Shared helpers for the TileLang LSTM recurrent-scan kernel tests
(``test_lstm_fwd`` / ``test_lstm_bwd`` / ``test_autotune``).

Centralises the SM80 skip marker, the seeded input builders, the per-band counter
allocation, the kernel-vs-reference tolerances/asserts, the common parametrize
matrices, and the end-to-end autograd reference + checks -- one source of truth
for what was copy-pasted across those files.

This module is imported only when the tilelstm tests are collected (which already
import the kernels), so it does not pull tilelang onto the non-tilelang collection
path the root ``conftest`` protects.
"""

import pytest
import torch

from caiman_asr_train.autotune.support import tilelang_supported
from caiman_asr_train.tilelstm import TileLangLSTM
from caiman_asr_train.utils.math import ceil_div

# SM80 gate: building/launching the recurrent-scan kernels needs cp.async
# (``T.async_copy``), unavailable below compute capability 8.0.
requires_tile_kernels = pytest.mark.skipif(
    not tilelang_supported(),
    reason=(
        "TileLang LSTM kernels need a CUDA GPU of compute capability >= 8.0 "
        "(Ampere+) and torch >= 2.7; cp.async (T.async_copy) is unavailable below "
        "SM80"
    ),
)

# Parametrize matrices shared by the forward and backward kernel-vs-ref checks.
H_TILES = [64, 128]  # the two config sizes
B_RAGGED = [1, 7, 40, 65, 100]  # batches that are not multiples of block_M=64

# Kernel-vs-reference tolerances (single source of truth for the magic numbers).
# Forward: Ys is fp16 off an fp32-internal scan (tight); Cs is the fp16-STORED saved
# cell compared against the fp32 reference down-cast, so only the saved-history
# rounding differs -- loosened to fp16 ULP.
FWD_Y_TOL = dict(atol=1e-3, rtol=1e-3)
FWD_C_TOL = dict(atol=2e-2, rtol=2e-2)
# Backward: one fp16-accumulation tolerance across all gate gradients.
BWD_TOL = dict(atol=1e-2, rtol=1e-2)


def assert_fwd_close(Ys, Cs, ref_ys, ref_cs):
    """Ys/Cs match the forward reference (Cs down-cast to its saved fp16)."""
    torch.testing.assert_close(Ys, ref_ys, **FWD_Y_TOL)
    torch.testing.assert_close(Cs, ref_cs.to(Cs.dtype), **FWD_C_TOL)


def assert_bwd_close(dGates, ref):
    """The kernel's (S+1)-slot dGates buffer matches the S-slot reference."""
    torch.testing.assert_close(dGates[: ref.shape[0]], ref, **BWD_TOL)


def assert_rel_close(actual, expected, tol, name):
    """
    Relative Frobenius-norm closeness. Robust to the fp16-vs-fp32 gap (and its
    compounding over layers) where an elementwise atol/rtol is brittle; a wrong
    shape/index/sign is still orders of magnitude off, not a few percent.
    """
    expected = expected.detach().float()
    rel = (
        actual.detach().float() - expected
    ).norm().item() / expected.norm().clamp_min(1e-12).item()
    assert rel < tol, f"{name}: relative error {rel:.3g} >= {tol}"


def band_counter(M, block_M, device, fpad=1):
    """
    The always-on per-band counter ``F`` (``[cdiv(M, block_M), fpad]`` int32, zeroed)
    every recurrent-scan kernel takes to drive its per-step sync. The test wrappers
    allocate it per call exactly as the autotuner/production wrapper does.
    """
    return torch.zeros(ceil_div(M, by=block_M), fpad, device=device, dtype=torch.int32)


# --- seeded input builders -------------------------------------------------
# These mirror ``autotune.buffers.scan_buffers`` / ``bwd_buffers`` (same shapes,
# the 1/H recurrent scaling, the NaN-filled output buffers that catch an unwritten
# slot) but seed an ISOLATED ``torch.Generator`` for reproducible failures and add
# the backward ``delta_scale`` knob the recurrent-gemm-isolation cases need.


def _seeded_rand(device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    return lambda *shape, dtype: torch.randn(  # noqa: E731
        *shape, device=device, dtype=dtype, generator=g
    )


def scan_inputs(H, B, S, device, seed=0):
    """
    The forward kernel's six call args ``(Y, C0, R, gates, Ys, Cs)``. ``R`` is scaled
    by 1/H so the fp16 recurrence stays bounded; ``C0`` is fp32 (the kernel keeps the
    running cell in fp32) while the rest are fp16; ``Ys``/``Cs`` are NaN-filled so a
    kernel that fails to write a slot is caught rather than read as stale zeros.
    """
    rand = _seeded_rand(device, seed)
    Y = rand(B, H, dtype=torch.float16)
    C0 = rand(B, H, dtype=torch.float32)
    R = rand(4, H, H, dtype=torch.float16) / H
    gates = rand(S, B, 4, H, dtype=torch.float16)
    Ys = torch.full((S + 1, B, H), float("nan"), device=device, dtype=torch.float16)
    Cs = torch.full((S + 1, B, H), float("nan"), device=device, dtype=torch.float16)
    return Y, C0, R, gates, Ys, Cs


def bwd_inputs(H, B, S, device, seed=0, delta_scale=1.0):
    """
    The backward kernel's inputs + its (S+1)-slot output buffer. ``As`` holds
    post-activation gates (sigmoid/tanh of random pre-activations -- realistic ranges
    keep fp16 well-conditioned); ``R`` is scaled by 1/H; ``dGates`` is NaN in slots
    0..S-1 (caught if unwritten) and zero in the pad slot S (read as the first reversed
    step's recurrent term). ``delta_scale=0.0`` zeroes ``delta`` to isolate the
    recurrent gemm. Returns ``(As, Cs, R, delta, dGates)``.
    """
    rand = _seeded_rand(device, seed)
    pre = rand(S, B, 4, H, dtype=torch.float32)
    As = torch.empty(S, B, 4, H, device=device, dtype=torch.float16)
    As[:, :, 0] = torch.sigmoid(pre[:, :, 0])
    As[:, :, 1] = torch.sigmoid(pre[:, :, 1])
    As[:, :, 2] = torch.tanh(pre[:, :, 2])
    As[:, :, 3] = torch.sigmoid(pre[:, :, 3])
    Cs = rand(S + 1, B, H, dtype=torch.float16)
    R = rand(4, H, H, dtype=torch.float16) / H
    delta = rand(S, B, H, dtype=torch.float16) * delta_scale
    dGates = torch.full(
        (S + 1, B, 4, H), float("nan"), device=device, dtype=torch.float16
    )
    dGates[S].zero_()  # the pad slot the first reversed step reads
    return As, Cs, R, delta, dGates


# --- end-to-end autograd reference + checks --------------------------------


def ref_lstm_layer(x, W_ih, W_hh, b_ih, b_hh, h0, c0):
    """
    fp32, fully-differentiable reference for one ``_TileLangLSTMLayer`` scan,
    mirroring the kernel's gate-major math (gate order i, f, g, o; recurrent term
    ``h @ R[k].T``). Returns the stacked hidden outputs ``[S, B, H]``.
    """
    S, B, _ = x.shape
    H = h0.shape[-1]
    gates = (x.reshape(S * B, -1) @ W_ih.t() + (b_ih + b_hh)).view(S, B, 4, H)
    R = W_hh.view(4, H, H)
    h, c = h0, c0
    outs = []
    for t in range(S):
        pre = torch.stack([h @ R[k].t() for k in range(4)], dim=1) + gates[t]
        i = torch.sigmoid(pre[:, 0])
        f = torch.sigmoid(pre[:, 1])
        g = torch.tanh(pre[:, 2])
        o = torch.sigmoid(pre[:, 3])
        c = f * c + i * g
        h = o * torch.tanh(c)
        outs.append(h)
    return torch.stack(outs, dim=0)


def check_end_to_end_grads(H=64, B=16, S=6):
    """
    The trainable ``TileLangLSTM`` layer (fp16 autocast: kernel forward saves
    activations, kernel backward produces dGates) yields param/input gradients
    matching a fp32 autograd reference of the same scan. This exercises the full
    wiring: dGates -> weight_ih/bias/x via the addmm autograd, dR -> weight_hh.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    model = TileLangLSTM(H, H, num_layers=1, device=device)
    layer = model.layers[0]

    x = torch.randn(S, B, H, device=device)
    target = torch.randn(S, B, H, device=device)
    h0 = torch.zeros(B, H, device=device)
    c0 = torch.zeros(B, H, device=device)

    # Kernel path: fp16 autocast, the trainable layer.
    x1 = x.clone().requires_grad_(True)
    with torch.autocast(device_type="cuda"):
        out1 = layer(x1, (h0, c0))
    (out1.float() * target).sum().backward()

    # fp32 reference path on cloned leaf params.
    W_ih = layer.weight_ih.detach().clone().requires_grad_(True)
    W_hh = layer.weight_hh.detach().clone().requires_grad_(True)
    b_ih = layer.bias_ih.detach().clone().requires_grad_(True)
    b_hh = layer.bias_hh.detach().clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)
    out2 = ref_lstm_layer(x2, W_ih, W_hh, b_ih, b_hh, h0, c0)
    (out2 * target).sum().backward()

    # fp16 kernel vs fp32 reference: loose tol absorbs the precision gap. A wrong
    # shape/index/sign in the wiring is orders of magnitude off, not a few %.
    tol = dict(atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out1.float(), out2, **tol)
    torch.testing.assert_close(x1.grad, x2.grad, **tol)
    torch.testing.assert_close(layer.weight_ih.grad, W_ih.grad, **tol)
    torch.testing.assert_close(layer.weight_hh.grad, W_hh.grad, **tol)
    torch.testing.assert_close(layer.bias_ih.grad, b_ih.grad, **tol)
    torch.testing.assert_close(layer.bias_hh.grad, b_hh.grad, **tol)


def check_multilayer_end_to_end_grads(L=3, H=64, B=16, S=6):
    """
    A multilayer ``TileLangLSTM`` (one shared hidden/cell allocation, per-layer kernel
    writes into slices) returns the CustomLSTM 3-tuple and yields param/input gradients
    matching a fp32 autograd reference that chains ``ref_lstm_layer`` across layers.
    Guards the multilayer autograd wiring (shared-buffer aliasing across the per-layer
    custom Functions) and the real (non-placeholder) ``c_n`` / ``all_c``.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    model = TileLangLSTM(H, H, num_layers=L, device=device)

    x = torch.randn(S, B, H, device=device)
    target = torch.randn(S, B, H, device=device)
    h0 = torch.zeros(B, H, device=device)
    c0 = torch.zeros(B, H, device=device)

    # Kernel path: fp16 autocast, full multilayer module (state=None -> zero seeds).
    x1 = x.clone().requires_grad_(True)
    with torch.autocast(device_type="cuda"):
        out1, (h_n, c_n), (all_h, all_c) = model(x1, None)
    (out1.float() * target).sum().backward()

    # The CustomLSTM return contract: shapes + a real (non-zero) final cell state.
    assert h_n.shape == (L, B, H) and c_n.shape == (L, B, H)
    assert all_h.shape == (L, S, B, H) and all_c.shape == (L, S, B, H)
    assert (
        c_n.abs().sum() > 0
    ), "c_n must be the real final cell, not a zero placeholder"

    # fp32 reference: chain the single-layer reference across layers on cloned leaves.
    x2 = x.clone().requires_grad_(True)
    ref_params = []
    cur = x2
    for layer in model.layers:
        W_ih = layer.weight_ih.detach().clone().requires_grad_(True)
        W_hh = layer.weight_hh.detach().clone().requires_grad_(True)
        b_ih = layer.bias_ih.detach().clone().requires_grad_(True)
        b_hh = layer.bias_hh.detach().clone().requires_grad_(True)
        ref_params.append((W_ih, W_hh, b_ih, b_hh))
        cur = ref_lstm_layer(cur, W_ih, W_hh, b_ih, b_hh, h0, c0)
    (cur * target).sum().backward()

    # Relative Frobenius norm per tensor (robust to the fp16-vs-fp32 gap compounding
    # over layers).
    tol = 5e-2
    assert_rel_close(out1.float(), cur, tol, "multilayer output")
    assert_rel_close(x1.grad, x2.grad, tol, "dx")
    for i, layer in enumerate(model.layers):
        W_ih, W_hh, b_ih, b_hh = ref_params[i]
        assert_rel_close(layer.weight_ih.grad, W_ih.grad, tol, f"layer {i} dW_ih")
        assert_rel_close(layer.weight_hh.grad, W_hh.grad, tol, f"layer {i} dW_hh")
        assert_rel_close(layer.bias_ih.grad, b_ih.grad, tol, f"layer {i} db_ih")
        assert_rel_close(layer.bias_hh.grad, b_hh.grad, tol, f"layer {i} db_hh")
