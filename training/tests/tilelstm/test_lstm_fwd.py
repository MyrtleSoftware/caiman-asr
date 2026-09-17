"""
Tests for the TileLang LSTM recurrent-scan kernels and their reference.

Two groups:

* ``lstm_ref`` -- the pure-PyTorch reference scan. CPU-runnable; checked for the
  in-place buffer contract (shapes, dtypes, seed slots) and against an
  independent ``torch.nn.LSTMCell`` so the gate-major layout/math is pinned.
* the two CUDA kernels (``lstm_coop_fwd``, ``lstm_persistent_fwd``) -- each built
  at a few tile configs and validated against ``lstm_ref`` on the same inputs,
  exactly as the autotuner validates its tuned winner. The full ``TileLangLSTM``
  module is also pinned against ``CustomLSTM`` (the implementation it drops in for).

Shared helpers (the SM80 marker, the seeded input builder, the per-band counter
allocation, and the kernel-vs-ref tolerances/asserts) live in ``_kernels``.
"""

from functools import partial

import pytest
import torch
from _kernels import (
    B_RAGGED,
    FWD_C_TOL,
    H_TILES,
    assert_fwd_close,
    assert_rel_close,
    band_counter,
    requires_tile_kernels,
    scan_inputs,
)

from caiman_asr_train.tilelstm import TileLangLSTM
from caiman_asr_train.tilelstm.lstm_fwd import (
    lstm_coop_fwd,
    lstm_persistent_fwd,
    lstm_ref,
)


def _band_counter_call(kernel, block_M: int, fpad: int, needs_cpp: bool = False):
    """Wrap a band-flag forward kernel's per-band counter F (zeroed per call) into the
    plain (Y, C0, R, gates, Ys, Cs) in-place contract, matching how the
    autotuner/production wrapper drives it. BOTH ``lstm_coop_fwd`` and
    ``lstm_persistent_fwd`` carry the always-on per-band counter F
    ([cdiv(M, block_M), fpad] int32) -- persistent's per-step grid barrier was replaced
    by the same band counter. The PERSISTENT forward also takes a 2-slot fp32 ping-pong
    cell channel ``Cpp`` ``[2, M, H]`` (the prior-cell re-read, kept full precision while
    ``Cs`` is stored fp16); ``needs_cpp`` injects it (mirrors ``wrap_coop_fwd_flags``).
    """

    def call(Y, C0, R, gates, Ys, Cs):
        M, H = Y.shape
        F = band_counter(M, block_M, Y.device, fpad)
        if needs_cpp:
            Cpp = torch.empty(2, M, H, device=Y.device, dtype=torch.float32)
            return kernel(Y, C0, R, gates, Ys, Cs, F, Cpp)
        return kernel(Y, C0, R, gates, Ys, Cs, F)

    return call


def _fwd_call(factory, H, **tile):
    """Build a forward kernel (``lstm_coop_fwd`` / ``lstm_persistent_fwd``) and wrap its
    per-band counter F (+ ping-pong Cpp for persistent) into the plain in-place contract
    (see ``_band_counter_call``)."""
    kernel = factory(H, **tile)
    needs_cpp = factory is lstm_persistent_fwd
    return _band_counter_call(
        kernel, tile.get("block_M", 64), tile.get("fpad", 1), needs_cpp
    )


# (backend builder, label) for the two recurrent-scan kernels. Both are called
# ``backend(H, **tile)`` and yield a callable on the (Y, C0, R, gates, Ys, Cs)
# in-place contract -- each via the F-allocating wrapper above (both now drive their
# per-step sync with the band counter F).
BACKENDS = [
    pytest.param(partial(_fwd_call, lstm_coop_fwd), id="coop"),
    pytest.param(partial(_fwd_call, lstm_persistent_fwd), id="persistent"),
]


def _tile(block_N: int = 32) -> dict:
    """
    Tile config for the kernel-vs-reference checks.

    ``block_N`` is capped at 32 for portability: each kernel stages a
    ``[block_M, 4*block_N]`` gate slab plus per-gate fp32 epilogue buffers in
    shared memory, so ``block_N=64`` needs ~128-136 KB/block. That fits an A100
    (164 KB opt-in) but exceeds the ~100 KB/block limit of 100KB-class GPUs
    (RTX 4090, A6000), where the launch aborts with "Failed to set the allowed
    dynamic shared memory size". (Production ``autotune._BLOCK_N`` now also carries
    48/64 for big-SMEM cards, but the generator's per-device SMEM filter prunes those
    on 100KB-class GPUs; this direct, UNFILTERED kernel test caps block_N at 32 so it
    stays runnable on every GPU we test on.)
    """
    return dict(block_M=64, block_N=block_N, block_K=64, num_stages=2, threads=128)


# ===== lstm_ref (pure PyTorch, CPU) =====


def test_lstm_ref_buffer_contract():
    """
    Shapes, dtypes and the seed slots (slot 0 is the (Y, C0) input).
    """
    H, B, S = 8, 3, 5
    Y, C0, R, gates, _, _ = scan_inputs(H, B, S, device="cpu")

    ys, cs = lstm_ref(Y, C0, R, gates)

    assert ys.shape == (S + 1, B, H) and ys.dtype == torch.float16
    assert cs.shape == (S + 1, B, H) and cs.dtype == torch.float32
    # Slot 0 is the seed, carried at the kernel's precision (hidden fp16, cell fp32).
    torch.testing.assert_close(ys[0], Y.half())
    torch.testing.assert_close(cs[0], C0.float())
    # The scan must populate every later slot (no NaNs leaking through).
    assert not torch.isnan(ys[1:]).any()
    assert not torch.isnan(cs[1:]).any()


def test_lstm_ref_matches_lstmcell():
    """
    Independent oracle: ``lstm_ref`` reproduces ``torch.nn.LSTMCell``.

    ``LSTMCell`` and the kernel share the (i, f, g, o) gate order. The kernel folds
    the input projection ``W_ih @ x + b_ih + b_hh`` into ``gates`` and keeps the
    recurrent weight as ``R`` ``[4, H, H] = weight_hh.view(4, H, H)``; feeding the
    cell those exact pieces makes the two scans the same computation, so they must
    agree. ``R``/``gates`` are cast to fp16 (``lstm_ref``'s contract -- the kernel
    operands are fp16); the loose tol absorbs that quantisation and ``lstm_ref``'s
    fp16 hidden state vs the cell's fp32.
    """
    torch.manual_seed(0)
    H, B, S = 16, 4, 4
    cell = torch.nn.LSTMCell(H, H)

    R = cell.weight_hh.detach().view(4, H, H).half()
    x = torch.randn(S, B, H)
    # gates[t] = W_ih @ x_t + b_ih + b_hh  -- both biases, since lstm_ref adds only
    # the recurrent term R @ h (no bias of its own). Cast to fp16 to match the
    # kernel operands lstm_ref expects.
    bias = cell.bias_ih.detach() + cell.bias_hh.detach()

    gates = (
        (x.reshape(S * B, H) @ cell.weight_ih.detach().t() + bias)
        .view(S, B, 4, H)
        .half()
    )

    # lstm_ref takes its precision from the input dtypes: seed the hidden in fp16
    # (lo, matching the fp16 R) and the cell in fp32 (hi). The fp32 cell runs from
    # its own fp32 zeros seed.
    ys, cs = lstm_ref(
        torch.zeros(B, H, dtype=torch.float16), torch.zeros(B, H), R, gates
    )

    h_t = torch.zeros(B, H)
    c_t = torch.zeros(B, H)
    for t in range(S):
        h_t, c_t = cell(x[t], (h_t, c_t))
        torch.testing.assert_close(ys[t + 1], h_t.half(), atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(cs[t + 1], c_t, atol=3e-2, rtol=3e-2)


# ===== CUDA kernels vs lstm_ref =====


@requires_tile_kernels
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("H", H_TILES)
def test_kernel_matches_ref(backend, H):
    """
    Each kernel's in-place scan matches the reference at the default tile.
    """
    device = torch.device("cuda")
    B, S = 64, 6
    Y, C0, R, gates, Ys, Cs = scan_inputs(H, B, S, device)

    kernel = backend(H, **_tile())
    # The kernel writes Ys/Cs in place; an out_idx=None JIT kernel returns no real
    # output (an empty result), so the answer is what landed in the buffers.
    out = kernel(Y, C0, R, gates, Ys, Cs)
    assert not out, "kernel is in-place and produces no output tensor"

    ref_ys, ref_cs = lstm_ref(Y, C0, R, gates)
    assert_fwd_close(Ys, Cs, ref_ys, ref_cs)


@requires_tile_kernels
@pytest.mark.parametrize("backend", BACKENDS)
def test_kernel_default_tile_launches(backend):
    """
    The factory's default tile builds a kernel that launches on this GPU.

    A bare ``backend(H)`` (no tile args) must fit the device's shared memory and
    match the reference. The default ``block_N`` is capped so 100KB-class GPUs
    (RTX 4090, A6000) can run it, not just A100-class (see ``lstm_fwd``); bumping
    the default back to a 128 KB+ tile would fail to launch here and trip this.
    """
    device = torch.device("cuda")
    H, B, S = 64, 64, 4  # H divisible by the default block_K=64
    Y, C0, R, gates, Ys, Cs = scan_inputs(H, B, S, device)

    kernel = backend(H)  # all-default tile config -- exercises the factory defaults
    kernel(Y, C0, R, gates, Ys, Cs)

    ref_ys, ref_cs = lstm_ref(Y, C0, R, gates)
    assert_fwd_close(Ys, Cs, ref_ys, ref_cs)


@requires_tile_kernels
@pytest.mark.parametrize("backend", BACKENDS)
def test_kernel_narrow_gate_tile(backend):
    """
    block_N=16 exercises the gate-major 4*block_N slab below the default width.
    """
    device = torch.device("cuda")
    H, B, S = 128, 64, 5
    Y, C0, R, gates, Ys, Cs = scan_inputs(H, B, S, device)

    kernel = backend(H, **_tile(block_N=16))
    kernel(Y, C0, R, gates, Ys, Cs)

    ref_ys, ref_cs = lstm_ref(Y, C0, R, gates)
    assert_fwd_close(Ys, Cs, ref_ys, ref_cs)


@requires_tile_kernels
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("B", B_RAGGED)
def test_kernel_ragged_batch(backend, B):
    """
    B not a multiple of block_M: partial batch tiles must compute correctly.

    TileLang's ``LegalizeSafeMemoryAccess`` pass predicates out-of-bound
    accesses for dynamic tensor extents, so all of these should produce the
    same output as ``lstm_ref``. The cases cover: B entirely below block_M
    (one partial tile), B just over block_M (one full + one very small tail),
    and a mid-range ragged case.
    """
    device = torch.device("cuda")
    H, S = 64, 4
    Y, C0, R, gates, Ys, Cs = scan_inputs(H, B, S, device)

    kernel = backend(H, **_tile())
    kernel(Y, C0, R, gates, Ys, Cs)

    ref_ys, ref_cs = lstm_ref(Y, C0, R, gates)
    assert_fwd_close(Ys, Cs, ref_ys, ref_cs)


@requires_tile_kernels
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("H", [48, 80])  # H % block_N(32) != 0, H % block_K(16) == 0
def test_kernel_ragged_block_n(backend, H):
    """
    H not a multiple of block_N: the column grid is ``cdiv(H, block_N)`` and TileLang
    predicates the ragged last column tile (the operand gather zero-fills the OOB rows,
    the global stores skip the OOB columns), so the scan still matches the reference. The
    band counter / persistent group size both count cdiv(H, block_N) column tiles.
    ``block_N`` stays a WMMA-n multiple (32) -- the factory asserts that; only the
    ``H % block_N`` divisibility is relaxed.
    """
    device = torch.device("cuda")
    B, S = 64, 5
    tile = dict(block_M=16, block_N=32, block_K=16, num_stages=2, threads=32)
    Y, C0, R, gates, Ys, Cs = scan_inputs(H, B, S, device)

    kernel = backend(H, **tile)
    kernel(Y, C0, R, gates, Ys, Cs)

    ref_ys, ref_cs = lstm_ref(Y, C0, R, gates)
    assert_fwd_close(Ys, Cs, ref_ys, ref_cs)


@requires_tile_kernels
@pytest.mark.parametrize("backend", BACKENDS)
def test_kernel_seeds_slot_zero(backend):
    """
    Slot 0 of Ys/Cs is the (Y0, C0) seed the kernel copies in before stepping.
    """
    device = torch.device("cuda")
    H, B, S = 64, 64, 3
    Y, C0, R, gates, Ys, Cs = scan_inputs(H, B, S, device)

    kernel = backend(H, **_tile())
    kernel(Y, C0, R, gates, Ys, Cs)

    torch.testing.assert_close(Ys[0], Y)
    # Cs is fp16; the seed is C0 down-cast.
    torch.testing.assert_close(Cs[0], C0.half())


@requires_tile_kernels
@pytest.mark.parametrize("H", H_TILES)
@pytest.mark.parametrize("block_N", [16, 32])
@pytest.mark.parametrize("blocks_per_sm,num_stages", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_persistent_matches_ref(H, block_N, blocks_per_sm, num_stages):
    """
    The FullRow register-epilogue persistent kernel matches the reference across
    ``blocks_per_sm`` (1 / 2) x ``num_stages`` x ``block_N``. The win is
    ``(bps=2, ns=2)`` -- pipelined gemm AND intra-SM overlap. Critically this is
    correct at BOTH block_N (16 and 32): the kernel drains the operand cp.async
    pipeline before the SMEM-merged Gs/Cp stores, so the ns>=2+2/SM alias race
    (Gs shares Ys's offset) cannot occur -- block_N=16 + (2,2) would silently
    corrupt without the drain.
    """
    device = torch.device("cuda")
    B, S = 64, 6
    Y, C0, R, gates, Ys, Cs = scan_inputs(H, B, S, device)

    kernel = lstm_persistent_fwd(
        H,
        blocks_per_sm=blocks_per_sm,
        block_M=64,
        block_N=block_N,
        block_K=64,
        num_stages=num_stages,
        threads=128,
    )
    _band_counter_call(kernel, block_M=64, fpad=1, needs_cpp=True)(
        Y, C0, R, gates, Ys, Cs
    )

    ref_ys, ref_cs = lstm_ref(Y, C0, R, gates)
    assert_fwd_close(Ys, Cs, ref_ys, ref_cs)


@requires_tile_kernels
@pytest.mark.parametrize(
    "H",
    [
        320,  # CT = 10: not a multiple of 8, gcd(10, 8) = 2
        384,  # CT = 12: not a multiple of 8, gcd(12, 8) = 4
        576,  # CT = 18: not a multiple of 8, gcd(18, 8) = 2
    ],
)
def test_persistent_column_group_coverage(H):
    """
    Regression: ``T.Persistent`` must cover EVERY (batch x hidden) tile even when
    the column-tile count ``CT = H // block_N`` is not a multiple of its default
    column group (8).

    ``T.Persistent`` blocks the column axis into groups of ``group_size`` (default
    8). When ``CT`` is not a multiple of that group AND exceeds it, the wave
    mapping over-counts the column axis (``ceil(CT/8)*8 > CT``) and a band of
    output tiles is NEVER written -- it reads back the NaN fill, the band growing
    with the batch. The kernel must pin ``group_size = gcd(CT, 8)`` (as
    ``lstm_persistent_bwd`` does) so the mapping is a bijection over all tiles.

    Pinned to ``blocks_per_sm=1`` (grid = #SMs): with the small grid ``RT * CT``
    spans several waves at ``B=2048`` (``RT=32`` row tiles), the regime where the
    gap appears. The existing small-batch checks (RT=1, CT<=8) cannot surface it.
    The explicit NaN assertions name the failure (a dropped tile) before the value
    comparison.
    """
    device = torch.device("cuda")
    B, S = 2048, 3  # RT = 32 row tiles -> RT*CT spans multiple grid waves
    Y, C0, R, gates, Ys, Cs = scan_inputs(H, B, S, device)

    kernel = lstm_persistent_fwd(H, blocks_per_sm=1, **_tile(block_N=32))
    _band_counter_call(kernel, block_M=64, fpad=1, needs_cpp=True)(
        Y, C0, R, gates, Ys, Cs
    )

    # A tile the wave mapping never visits keeps its NaN seed -- catch that
    # directly so the failure mode is unambiguous, not just an allclose mismatch.
    assert not torch.isnan(Ys[1:]).any(), "persistent fwd left hidden tiles unwritten"
    assert not torch.isnan(Cs[1:]).any(), "persistent fwd left cell tiles unwritten"

    ref_ys, ref_cs = lstm_ref(Y, C0, R, gates)
    assert_fwd_close(Ys, Cs, ref_ys, ref_cs)


@requires_tile_kernels
@pytest.mark.parametrize("block_N", [8, 16])
def test_persistent_merge_prefetch_no_race(block_N):
    """
    Regression: the SMEM-merge prefetch must not clobber operands the gemm is
    still reading.

    The merge aliases ``Gs_shared``/``Cp_shared`` onto the now-dead operand tiles
    (``Ys_shared``/``rW_shared``). Without a block barrier between the gemm K-loop
    and the ``cp.async`` prefetch, the prefetch writes those shared addresses while
    other warps are still issuing ``ldmatrix`` reads of the operands -- a
    cross-warp WAR (``cp_async_gs`` WRITE vs ``ptx_ldmatrix_x4`` READ under
    racecheck). It only bites with enough SM contention: ``blocks_per_sm=2`` +
    ``num_stages>=2`` + small ``block_N``, and a batch big enough to fill the grid
    -- exactly the gate-saving training-forward winner at H=1024 (block_N=16,
    bps=2). The scattered NaN it leaves is caught here against the reference.

    ``block_N`` 8 and 16 are the small widths that expose it; 32 was always clean.
    """
    device = torch.device("cuda")
    H, B, S = 1024, 256, 4
    Y, C0, R, gates, Ys, Cs = scan_inputs(H, B, S, device)

    kernel = lstm_persistent_fwd(
        H,
        block_M=64,
        block_N=block_N,
        block_K=64,
        num_stages=2,
        threads=128,
        blocks_per_sm=2,
    )
    _band_counter_call(kernel, block_M=64, fpad=1, needs_cpp=True)(
        Y, C0, R, gates, Ys, Cs
    )

    assert not torch.isnan(Ys[1:]).any(), "persistent fwd prefetch clobbered operands"
    assert not torch.isnan(Cs[1:]).any(), "persistent fwd prefetch clobbered operands"

    ref_ys, ref_cs = lstm_ref(Y, C0, R, gates)
    # Ys is loosened to 1e-2 here (vs the usual 1e-3): the H=1024 recurrence over
    # this larger grid accumulates more fp16 rounding than the small-H scans.
    torch.testing.assert_close(Ys, ref_ys, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(Cs, ref_cs.to(Cs.dtype), **FWD_C_TOL)


# ===== Full module vs CustomLSTM (the implementation it drops in for) =====


@requires_tile_kernels
def test_tilelang_matches_custom_lstm_forward(cap_autotune_configs):
    """
    ``TileLangLSTM`` is a drop-in for ``CustomLSTM``: with the SAME weights and input,
    the inference (no-grad) forward must produce a matching output and final state.
    Pins the whole module/kernel/autotune stack to an independent, already-trusted
    implementation -- stronger than the internal ``lstm_ref`` checks above.
    """
    from rnnt_ext.custom_lstm.lstm import CustomLSTM

    torch.manual_seed(0)
    device = torch.device("cuda")
    H, B, S, L = 64, 32, 6, 2
    custom = CustomLSTM(H, H, num_layers=L, device=device)
    tl = TileLangLSTM(H, H, num_layers=L, device=device)
    tl.load_state_dict(custom.state_dict(), strict=True)  # identical weights

    x = torch.randn(S, B, H, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda"):
        out_c, (h_c, c_c), _ = custom(x)
        out_t, (h_t, c_t), _ = tl(x)

    assert_rel_close(out_t, out_c, 5e-2, "output")
    assert_rel_close(h_t, h_c, 5e-2, "h_n")
    assert_rel_close(c_t, c_c, 5e-2, "c_n")
