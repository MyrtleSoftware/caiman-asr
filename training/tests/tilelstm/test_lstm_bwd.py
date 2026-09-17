"""
Tests for the TileLang LSTM recurrent-scan BACKWARD kernel and its reference.

Two groups, mirroring ``test_lstm_fwd.py``:

* ``lstm_bwd_ref`` -- the pure-PyTorch reference reverse-scan. CPU-runnable;
  checked for its buffer contract and, independently, against autograd
  back-propagated through ``lstm_ref`` (so the gate-gradient math is pinned to
  the forward, not just to the legacy CUDA transcription).
* the CUDA kernel (``lstm_coop_bwd``) -- built at a few tile configs and validated
  against ``lstm_bwd_ref`` on the same inputs, exactly as the autotuner validates
  its tuned winner. A ``delta = 0`` case isolates the recurrent gemm (the
  no-transpose, middle-axis contraction that is the load-bearing departure from
  the forward).

The full ``TileLangLSTM`` module's forward+backward is also pinned end-to-end:
against a fp32 autograd reference and against ``CustomLSTM`` (the implementation it
drops in for). Shared helpers (the SM80 marker, the seeded input builder, the
per-band counter, the tolerances/asserts, and the end-to-end checks) live in
``_kernels``.
"""

import pytest
import torch
from _kernels import (
    B_RAGGED,
    H_TILES,
    assert_bwd_close,
    assert_fwd_close,
    assert_rel_close,
    band_counter,
    bwd_inputs,
    check_end_to_end_grads,
    check_multilayer_end_to_end_grads,
    requires_tile_kernels,
    scan_inputs,
)
from tilelang.language import GemmWarpPolicy

from caiman_asr_train.tilelstm import TileLangLSTM
from caiman_asr_train.tilelstm.autotune.wrappers import (
    wrap_coop_fwd_flags,
    wrap_splitk_bwd,
)
from caiman_asr_train.tilelstm.lstm_bwd import (
    lstm_bwd_ref,
    lstm_coop_bwd,
    lstm_coop_bwd_splitk,
    lstm_persistent_bwd,
)
from caiman_asr_train.tilelstm.lstm_fwd import (
    lstm_coop_fwd,
    lstm_persistent_fwd,
    lstm_ref,
)

# ===== lstm_bwd_ref (pure PyTorch, CPU) =====


def test_lstm_bwd_ref_buffer_contract():
    """
    Shapes and dtypes; the reference omits the kernel's (S+1)-th pad slot.
    """
    H, B, S = 8, 3, 5
    As, Cs, R, delta, _ = bwd_inputs(H, B, S, device="cpu")

    dgates = lstm_bwd_ref(As, Cs, R, delta)

    assert dgates.shape == (S, B, 4, H)
    assert dgates.dtype == torch.float16
    assert not torch.isnan(dgates).any()


def test_lstm_bwd_ref_matches_autograd():
    """
    Independent oracle: ``lstm_bwd_ref`` reproduces autograd through ``lstm_ref``.

    ``lstm_ref``'s gate pre-activation is ``pre[t] = Ys[t] @ R.T + gates[t]``;
    since ``gates[t]`` enters the loss only through ``pre[t]``, the autograd grad
    ``d loss / d gates`` IS ``dGates`` (grad of the pre-activations). Run the whole
    check in fp32 (seed/inputs fp32 -> ``lstm_ref`` runs fp32) so the comparison
    pins the derivative *math* without fp16 noise. ``As`` is recomputed from the
    same forward ``Ys``/``gates``/``R`` so it is consistent with ``cs``.
    """
    torch.manual_seed(0)
    H, B, S = 12, 4, 6
    Y = torch.randn(B, H, dtype=torch.float32)
    C0 = torch.randn(B, H, dtype=torch.float32)
    R = torch.randn(4, H, H, dtype=torch.float32) / H
    gates = torch.randn(S, B, 4, H, dtype=torch.float32, requires_grad=True)
    delta = torch.randn(S, B, H, dtype=torch.float32)

    ys, cs = lstm_ref(Y, C0, R, gates)
    (ys[1:] * delta).sum().backward()

    # Recompute the post-activation gates from the (detached) forward states, the
    # same computation lstm_ref performed internally.
    ys_d, g_d = ys.detach(), gates.detach()
    pre = torch.stack([ys_d[:-1] @ R[k].T for k in range(4)], dim=2) + g_d
    As = torch.stack(
        [
            torch.sigmoid(pre[:, :, 0]),
            torch.sigmoid(pre[:, :, 1]),
            torch.tanh(pre[:, :, 2]),
            torch.sigmoid(pre[:, :, 3]),
        ],
        dim=2,
    )

    ref = lstm_bwd_ref(As, cs.detach(), R, delta)
    torch.testing.assert_close(ref, gates.grad, atol=1e-4, rtol=1e-4)


# ===== CUDA kernel vs lstm_bwd_ref =====


def _coop_bwd_call(
    H,
    block_M=64,
    block_N=32,
    block_K=32,
    num_stages=2,
    threads=128,
    policy=GemmWarpPolicy.Square,
    wpf=False,
    fpad=1,
):
    """Build the coop backward kernel and wrap its per-band counter F (zeroed per
    call) into the 5-arg contract, matching how the autotuner/production wrapper
    drives it (the band counters are always-on; F is [cdiv(M,block_M), fpad] int32).
    ``policy`` ("square"/"fullcol") and ``wpf`` (As wait-prefetch) are the kernel's
    build-mode toggles, threaded through so tests can exercise each one."""
    kernel = lstm_coop_bwd(
        H,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_stages=num_stages,
        threads=threads,
        policy=policy,
        wpf=wpf,
        fpad=fpad,
    )

    def call(As, Cs, R, delta, dGates):
        F = band_counter(As.shape[1], block_M, As.device, fpad)
        return kernel(As, Cs, R, delta, dGates, F)

    return call


@requires_tile_kernels
@pytest.mark.parametrize("H", H_TILES)
def test_bwd_kernel_matches_ref(H):
    """
    The in-place reverse scan matches the reference at the default tile.
    """
    device = torch.device("cuda")
    B, S = 64, 6
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device)

    out = _coop_bwd_call(
        H, block_M=64, block_N=32, block_K=32, num_stages=2, threads=128
    )(As, Cs, R, delta, dGates)
    assert not out, "kernel is in-place and produces no output tensor"

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


@requires_tile_kernels
def test_bwd_kernel_narrow_gate_tile():
    """
    block_N=16 exercises the gate-major slabs at the narrowest productive width.
    """
    device = torch.device("cuda")
    H, B, S = 128, 64, 5
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device)

    _coop_bwd_call(H, block_M=64, block_N=16, block_K=64, num_stages=1, threads=128)(
        As, Cs, R, delta, dGates
    )

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


@requires_tile_kernels
def test_bwd_kernel_isolates_recurrent_gemm():
    """
    delta = 0: dh comes purely from the recurrent term dGates[t+1] @ rW, so this
    isolates the no-transpose, middle-axis contraction (the load-bearing
    departure from the forward gemm). A wrong rW axis fails here but might pass
    the full test where delta dominates.
    """
    device = torch.device("cuda")
    H, B, S = 64, 32, 5
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device, delta_scale=0.0)
    assert (delta == 0).all()

    _coop_bwd_call(H, block_M=64, block_N=32, block_K=32, num_stages=2, threads=128)(
        As, Cs, R, delta, dGates
    )

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


@requires_tile_kernels
@pytest.mark.parametrize("B", [96, 100])  # neither a multiple of block_M=64
@pytest.mark.parametrize(
    "H,block_N,block_K,num_stages",
    [
        (64, 32, 32, 2),  # the default-shaped tile
        (128, 16, 64, 3),  # narrow gate slab, fat K-chunk, deeper pipeline
    ],
)
def test_bwd_kernel_wpf(H, block_N, block_K, num_stages, B):
    """
    wpf=True (wait-prefetch): the four post-activation gate slabs of ``As[t_rev]``
    are staged into SMEM (``As_shared``) ahead of the recurrent gemm and drained
    with ``ptx_wait_group(0)`` before the gate epilogue reads them, instead of the
    default direct-from-global reads every other coop test uses. This is the only
    coverage of that prefetch and its drain; a missing/early wait would feed the
    epilogue stale ``As`` and is caught here. Two tile configs x two batch sizes
    that are not multiples of block_M (the last m-tile is ragged while the
    prefetched ``As_shared`` is live across the gemm).
    """
    device = torch.device("cuda")
    S = 5
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device)

    _coop_bwd_call(
        H,
        block_M=64,
        block_N=block_N,
        block_K=block_K,
        num_stages=num_stages,
        threads=128,
        wpf=True,
    )(As, Cs, R, delta, dGates)

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


@requires_tile_kernels
@pytest.mark.parametrize("B", [64, 100])  # neither a multiple of block_M=48
@pytest.mark.parametrize("H", H_TILES)
def test_bwd_kernel_fullcol(H, B):
    """
    policy="fullcol" + block_M=48: the non-power-of-two batch tile the autotuner
    uses for grid-fill. ``coop_bwd_configs`` tags ``block_M % 32`` as fullcol --
    Square's balanced 2-warp M split asserts ``16 | block_M/2``, which 48 fails, so
    it must run under the FullCol (m_warp=1) partition. Exercises that warp policy
    AND the ragged last m-tile predication (48 divides neither batch). Two H
    (config sizes) x two batch sizes that are not multiples of block_M.
    """
    device = torch.device("cuda")
    S = 5
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device)

    _coop_bwd_call(
        H,
        block_M=48,
        block_N=32,
        block_K=64,
        num_stages=2,
        threads=128,
        policy=GemmWarpPolicy.FullCol,
    )(As, Cs, R, delta, dGates)

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


# ===== Forward save_gates: the activations the backward consumes =====


@requires_tile_kernels
@pytest.mark.parametrize("H", H_TILES)
def test_coop_fwd_save_gates_emits_activations(H):
    """
    ``lstm_coop_fwd(save_gates=True)`` overwrites ``gates`` IN PLACE with the
    post-activation gates the backward reads (As aliases Gs), with the ``Ys``/``Cs``
    outputs unchanged.

    The reference activations are ``act(Ys[t] @ R[k].T + gates[t])`` recomputed from
    the reference's own hidden states, the same pre-activation ``lstm_ref`` forms
    internally (so the comparison is exact up to fp16). Because the kernel overwrites
    ``gates``, the reference is built from a pristine clone taken before the call.
    """
    device = torch.device("cuda")
    B, S = 64, 6
    Y, C0, R, gates, Ys, Cs = scan_inputs(H, B, S, device, seed=1)
    gates_in = gates.clone()  # pristine projection; the kernel overwrites `gates`

    kernel = lstm_coop_fwd(
        H,
        block_M=64,
        block_N=32,
        block_K=64,
        num_stages=2,
        threads=128,
        save_gates=True,
    )
    # coop_fwd drives its per-step sync with the per-band counter F (always-on):
    # (Y, C0, R, gates, Ys, Cs, F). With save_gates it overwrites ``gates`` with the
    # post-activation gates in place (no separate As arg).
    F = band_counter(B, 64, device)
    kernel(Y, C0, R, gates, Ys, Cs, F)
    As = gates  # the activations the kernel wrote in place

    ref_ys, ref_cs = lstm_ref(Y, C0, R, gates_in)
    assert_fwd_close(Ys, Cs, ref_ys, ref_cs)

    As_ref = torch.empty(S, B, 4, H, device=device, dtype=torch.float16)
    for t in range(S):
        pre = (
            torch.stack(
                [
                    torch.mm(ref_ys[t], R[k].T, out_dtype=torch.float32)
                    for k in range(4)
                ],
                dim=1,
            )
            + gates_in[t].float()
        )
        As_ref[t, :, 0] = torch.sigmoid(pre[:, 0]).half()
        As_ref[t, :, 1] = torch.sigmoid(pre[:, 1]).half()
        As_ref[t, :, 2] = torch.tanh(pre[:, 2]).half()
        As_ref[t, :, 3] = torch.sigmoid(pre[:, 3]).half()
    torch.testing.assert_close(As, As_ref, atol=1e-2, rtol=1e-2)


@requires_tile_kernels
@pytest.mark.parametrize("B", B_RAGGED)
def test_bwd_kernel_ragged_batch(B):
    """
    B not a multiple of block_M: partial batch tiles must compute correctly.
    """
    device = torch.device("cuda")
    H, S = 64, 4
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device)

    _coop_bwd_call(H, block_M=64, block_N=32, block_K=32, num_stages=2, threads=128)(
        As, Cs, R, delta, dGates
    )

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


# ===== Persistent backward kernel (vs lstm_bwd_ref) =====


def _persistent_bwd_call(
    H,
    block_M=64,
    block_N=32,
    block_K=32,
    num_stages=2,
    threads=128,
    blocks_per_sm=1,
    policy=GemmWarpPolicy.Square,
):
    """Build the persistent backward kernel and wrap its (M,H) dc_fut scratch + per-band
    counter F away, matching the 5-arg coop call contract (the kernel seeds dC itself; F
    is zeroed per call -- the kernel drives its per-step sync with the band flags, like
    the persistent forward). ``policy`` is the recurrent-gemm warp partition (Square
    default, FullCol for odd-16 block_M)."""
    kernel = lstm_persistent_bwd(
        H,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_stages=num_stages,
        threads=threads,
        blocks_per_sm=blocks_per_sm,
        policy=policy,
    )

    def call(As, Cs, R, delta, dGates):
        M = As.shape[1]
        # fp32 carry (the kernel's dtype_hi), NOT Cs.dtype -- Cs is now fp16.
        dC = torch.empty(M, As.shape[3], device=As.device, dtype=torch.float32)
        F = band_counter(M, block_M, As.device)
        kernel(As, Cs, R, delta, dGates, dC, F)

    return call


@requires_tile_kernels
@pytest.mark.parametrize("H", H_TILES)
def test_persistent_bwd_kernel_matches_ref(H):
    """
    The persistent reverse scan (fixed sm_num grid, global dc_fut carry) matches the
    reference -- the carry round-trips through global correctly across the per-step
    grid barrier.
    """
    device = torch.device("cuda")
    B, S = 96, 6
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device)

    _persistent_bwd_call(H)(As, Cs, R, delta, dGates)

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


@requires_tile_kernels
def test_persistent_bwd_kernel_blocks_per_sm_2():
    """
    blocks_per_sm=2 (two co-resident blocks). The backward's gate-major operands are
    heavy, so 2/SM only fits a small tile (bM=32,bN=16,bK=32 -> ~50KB); a bigger tile
    is excluded by _persistent_bwd_configs, so this is the feasible bps=2 shape.
    """
    device = torch.device("cuda")
    H, B, S = 128, 96, 5
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device)

    _persistent_bwd_call(
        H, block_M=32, block_N=16, block_K=32, num_stages=2, blocks_per_sm=2
    )(As, Cs, R, delta, dGates)

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


@requires_tile_kernels
def test_persistent_bwd_kernel_narrow_gate_tile():
    """block_N=16 + num_stages>=2 exercises the pipelined gate-major operand."""
    device = torch.device("cuda")
    H, B, S = 128, 80, 5
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device)

    _persistent_bwd_call(H, block_N=16, block_K=32, num_stages=2)(
        As, Cs, R, delta, dGates
    )

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


@requires_tile_kernels
def test_persistent_bwd_kernel_isolates_recurrent_gemm():
    """delta = 0: dh is purely the recurrent term, isolating the gemm + the global
    dc_fut carry (a wrong carry index/sign fails here even when delta would mask it)."""
    device = torch.device("cuda")
    H, B, S = 64, 48, 5
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device, delta_scale=0.0)
    assert (delta == 0).all()

    _persistent_bwd_call(H)(As, Cs, R, delta, dGates)

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


@requires_tile_kernels
@pytest.mark.parametrize("B", [1, 7, 100])
def test_persistent_bwd_kernel_ragged_batch(B):
    """B not a multiple of block_M: partial batch tiles (and idle persistent blocks
    when num_tiles < grid) must still compute correctly."""
    device = torch.device("cuda")
    H, S = 64, 4
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device)

    _persistent_bwd_call(H)(As, Cs, R, delta, dGates)

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


# ===== Split-K backward kernel (vs lstm_bwd_ref) =====


def _splitk_bwd_call(
    H,
    k_splits=4,
    block_M=64,
    block_N=32,
    block_K=64,
    num_stages=2,
    threads=128,
    red_M=4,
    res_kc=None,
):
    """Build the split-K backward kernel behind the production wrapper (which
    allocates the P/dC scratches), matching the 5-arg call contract. ``res_kc``
    (resident rW-strip prefix; None = full strip) is threaded through so tests can
    exercise the streamed-tail mode the production sweep selects at large H."""
    kernel = lstm_coop_bwd_splitk(
        H,
        k_splits=k_splits,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_stages=num_stages,
        threads=threads,
        red_M=red_M,
        res_kc=res_kc,
    )
    return wrap_splitk_bwd(kernel, {"k_splits": k_splits, "red_M": red_M})


@requires_tile_kernels
@pytest.mark.parametrize("H", H_TILES)
def test_splitk_bwd_kernel_matches_ref(H):
    """
    The two-phase split-K reverse scan matches the reference at an
    exact-divide batch (the historical contract: B % (k_splits * red_M) == 0).
    """
    device = torch.device("cuda")
    B, S = 64, 6
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device)

    _splitk_bwd_call(H)(As, Cs, R, delta, dGates)

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


@requires_tile_kernels
@pytest.mark.parametrize("B", [1, 2, 3, 7, 40, 65, 100])
def test_splitk_bwd_kernel_ragged_batch(B):
    """
    B not a multiple of k_splits * red_M: the red_M-rounded cdiv row shares
    must predicate the rows >= B (including ENTIRELY empty trailing blocks at
    very small B < k_splits * red_M, e.g. B=1/2/3/7) -- the kernel is dynamic in
    batch, so a ragged last dataloader batch must work with the tuned-at-a-bucket
    build. B=1/2/3 also pin the small-batch contract that splitk_bwd_configs now
    relies on when it emits configs all the way down to B=1 (block_M=64 > B is a
    fully-ragged single m-tile; red_M does not divide the row share).
    """
    device = torch.device("cuda")
    H, S = 64, 4
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device)

    _splitk_bwd_call(H)(As, Cs, R, delta, dGates)

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


@requires_tile_kernels
@pytest.mark.parametrize("block_M", [16, 32])
@pytest.mark.parametrize("B", [1, 2, 3, 16, 48])
def test_splitk_bwd_kernel_small_block_m(block_M, B):
    """
    The pad-reducing small m-tiles (16/32 < the 64 default) that splitk_bwd_configs
    emits for B < 64: block_M is the phase-1 gemm m-dim (must be a multiple of the
    16-row MMA atom), so 16/32 cut the phase-1 padding a 64-tile would carry at tiny
    batch. Verify they match the reference across small batches (B < block_M, B between
    block_M and 64, and B = block_M).
    """
    device = torch.device("cuda")
    H, S = 64, 4
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device)

    _splitk_bwd_call(H, k_splits=4, block_M=block_M, block_N=32, block_K=64, red_M=4)(
        As, Cs, R, delta, dGates
    )

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


@requires_tile_kernels
@pytest.mark.parametrize("B", [16, 100])
def test_splitk_bwd_wrapper_dynamic_batch(B):
    """
    The production wrapper imposes no batch constraint: a kernel tuned at one
    bucket runs at any other batch through the 5-arg contract (this is what
    select_fastest's bench-at-real-B does after tuning at the bucket; a raise
    here used to crash selection at ragged batches).
    """
    device = torch.device("cuda")
    H, S = 64, 4
    call = _splitk_bwd_call(H, k_splits=4, red_M=8)  # ks*red_M = 32, B % 32 != 0
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device)
    call(As, Cs, R, delta, dGates)
    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


@requires_tile_kernels
@pytest.mark.parametrize("B", [65, 100])  # neither a multiple of block_M=64
@pytest.mark.parametrize(
    "H,block_K,res_kc",
    [
        (64, 32, 1),  # k_range=64, kchunks=2 -> tail=1 streamed K-chunk
        (128, 32, 2),  # k_range=128, kchunks=4 -> tail=2 streamed K-chunks
    ],
)
def test_splitk_bwd_partial_resident(H, block_K, res_kc, B):
    """
    res_kc < kchunks: only the first ``res_kc`` K-chunks of the block's rW strip
    stay resident; the remaining ``tail`` chunks are streamed per m-tile through
    the pipelined ``rW_pipe`` (the kernel's ``if tail:`` gemm loop). Every other
    split-K test uses the default ``res_kc=None`` (full strip, tail=0), so this is
    the ONLY coverage of the streamed-tail gemm -- the partial-resident mode the
    H>=1536 sweep selects as the production winner. Two configs (1 and 2 streamed
    tail chunks) x two batch sizes that are not multiples of block_M.
    """
    device = torch.device("cuda")
    S = 5
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device)

    _splitk_bwd_call(
        H,
        k_splits=4,
        block_M=64,
        block_N=32,
        block_K=block_K,
        num_stages=2,
        threads=128,
        red_M=4,
        res_kc=res_kc,
    )(As, Cs, R, delta, dGates)

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


# ===== End-to-end autograd =====


@requires_tile_kernels
def test_tilelang_lstm_end_to_end_grads(cap_autotune_configs):
    """Auto kernel selection (coop at this shape)."""
    check_end_to_end_grads()


@requires_tile_kernels
def test_tilelang_lstm_end_to_end_grads_persistent(monkeypatch):
    """
    Same end-to-end gradient check, but the layer's training forward AND backward
    are forced to the PERSISTENT kernels (which auto-selection only reaches past the
    residency wall, i.e. at large H/B that are too slow for a unit test). This
    gradient-checks the persistent backward's full autograd wiring at a small shape:
    the persistent forward must save the activations the persistent backward reads,
    and the wrapped dc_fut scratch must round-trip correctly.
    """
    import caiman_asr_train.tilelstm.lstm as lstm_mod

    H = 64
    # The persistent forward drives its per-step sync with the band counter F (ported
    # from coop) and carries the fp32 ping-pong cell channel Cpp, so wrap it exactly as
    # the tuner's finalize does -- the wrapper allocates+zeroes F and Cpp per call and
    # presents the layer's training contract (Y0, C0, R, gates, Ys, Cs); the gate-saving
    # forward writes its activations back into ``gates`` in place. ``blocks_per_sm`` in
    # the cfg is what selects the persistent (Cpp-injecting) path in the wrapper.
    fwd = wrap_coop_fwd_flags(
        lstm_persistent_fwd(
            H,
            block_M=64,
            block_N=32,
            block_K=64,
            num_stages=2,
            threads=128,
            blocks_per_sm=1,
            save_gates=True,
        ),
        {"block_M": 64, "fpad": 1, "blocks_per_sm": 1},
    )
    bwd_raw = lstm_persistent_bwd(
        H,
        block_M=64,
        block_N=32,
        block_K=32,
        num_stages=2,
        threads=128,
        blocks_per_sm=1,
    )

    def bwd_call(As, Cs, R, delta, dGates):
        M = As.shape[1]
        # fp32 carry (the kernel's dtype_hi), NOT Cs.dtype -- Cs is now fp16.
        dC = torch.empty(M, As.shape[3], device=As.device, dtype=torch.float32)
        F = band_counter(M, 64, As.device)
        bwd_raw(As, Cs, R, delta, dGates, dC, F)

    monkeypatch.setattr(lstm_mod, "select_fwd_train_kernel", lambda *a, **k: fwd)
    monkeypatch.setattr(lstm_mod, "select_bwd_train_kernel", lambda *a, **k: bwd_call)

    check_end_to_end_grads(H=H)


@requires_tile_kernels
def test_tilelang_lstm_multilayer_end_to_end_grads(cap_autotune_configs):
    """3-layer end-to-end: the CustomLSTM 3-tuple return, a real ``c_n``, and multilayer
    autograd through the single shared hidden/cell buffer."""
    check_multilayer_end_to_end_grads()


@requires_tile_kernels
def test_tilelang_matches_custom_lstm(cap_autotune_configs):
    """
    Drop-in parity with the trusted CUDA ``CustomLSTM`` under training (grad) mode:
    with shared weights and input, the forward outputs AND every parameter / input
    gradient match. The strongest behavioral guard -- it pins the full stack (module
    wiring, fwd+bwd kernels, autotune selection) against an independent implementation,
    not just the internal ``lstm_ref``.
    """
    from rnnt_ext.custom_lstm.lstm import CustomLSTM

    torch.manual_seed(0)
    device = torch.device("cuda")
    H, B, S, L = 64, 16, 6, 2
    custom = CustomLSTM(H, H, num_layers=L, device=device)
    tl = TileLangLSTM(H, H, num_layers=L, device=device)
    tl.load_state_dict(custom.state_dict(), strict=True)  # identical weights

    target = torch.randn(S, B, H, device=device)

    def run(model, x):
        with torch.autocast(device_type="cuda"):
            out = model(x)[0]
        (out.float() * target).sum().backward()
        return out

    xc = torch.randn(S, B, H, device=device, requires_grad=True)
    xt = xc.detach().clone().requires_grad_(True)
    out_c = run(custom, xc)
    out_t = run(tl, xt)

    assert_rel_close(out_t, out_c, 5e-2, "output")
    assert_rel_close(xt.grad, xc.grad, 5e-2, "dx")
    for i in range(L):
        for w in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            assert_rel_close(
                getattr(tl, f"{w}_l{i}").grad,
                getattr(custom, f"{w}_l{i}").grad,
                5e-2,
                f"layer {i} {w}",
            )


# ===== Ragged-N (H not a multiple of block_N) across the three backward kernels =====
# The column grid is cdiv(H, block_N); TileLang predicates the ragged last column tile
# (operand gather zero-fills, stores skip OOB). block_N stays a WMMA-n multiple (48);
# only the H % block_N divisibility is relaxed. block_K still divides H.


@requires_tile_kernels
@pytest.mark.parametrize("H", [64, 80])  # H % block_N(48) != 0, H % block_K(16) == 0
def test_bwd_kernel_ragged_block_n(H):
    device = torch.device("cuda")
    B, S = 64, 6
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device)

    _coop_bwd_call(H, block_M=64, block_N=48, block_K=16, num_stages=2, threads=128)(
        As, Cs, R, delta, dGates
    )

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


@requires_tile_kernels
@pytest.mark.parametrize("H", [64, 80])
def test_persistent_bwd_kernel_ragged_block_n(H):
    device = torch.device("cuda")
    B, S = 96, 6
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device)

    _persistent_bwd_call(
        H, block_M=64, block_N=48, block_K=16, num_stages=2, threads=128
    )(As, Cs, R, delta, dGates)

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


@requires_tile_kernels
def test_persistent_bwd_kernel_odd16_block_m():
    """persistent_bwd runs the default Square gemm; an odd-16 block_M (16, NOT a multiple
    of 32) is legal where Square balances to m_warp=1 (wide block_N, the FullCol-
    equivalent split) and computes correctly. So the config filter gates on
    ``_wmma_ok(Square)`` + SMEM, not a block_M % 32 rule (block_M=48 is the production
    case -- same m_warp=1 regime, but its wide-bN epilogue only fits A100-class SMEM).
    """
    device = torch.device("cuda")
    H, B, S = 64, 64, 5  # block_N=64 -> Square(16,64,4) picks m_warp=1
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device)

    _persistent_bwd_call(
        H, block_M=16, block_N=64, block_K=64, num_stages=2, threads=128
    )(As, Cs, R, delta, dGates)

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


@requires_tile_kernels
def test_persistent_bwd_kernel_fullcol():
    """persistent_bwd now takes a ``policy`` arg: an odd-16 ``block_M`` (48) at NARROW
    ``block_N`` (32) is warp-illegal under Square (balanced -> m_warp=2, 48 % 32 != 0) but
    legal under FullCol (m_warp=1, 48 % 16 == 0). It must compute correctly -- the gemm
    result is policy-invariant and the SMEM-merged epilogue reads the logical [bM, bN] dh.
    This is the tile the FullCol expansion unlocks for the persistent grid.
    """
    device = torch.device("cuda")
    H, B, S = 64, 64, 5
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device)

    _persistent_bwd_call(
        H,
        block_M=48,
        block_N=32,
        block_K=32,
        num_stages=2,
        threads=128,
        policy=GemmWarpPolicy.FullCol,
    )(As, Cs, R, delta, dGates)

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)


@requires_tile_kernels
def test_splitk_bwd_kernel_ragged_block_n():
    device = torch.device("cuda")
    H, B, S = 48, 64, 6  # H % block_N(32) != 0 (cdiv=2), H % block_K(16) == 0
    As, Cs, R, delta, dGates = bwd_inputs(H, B, S, device)

    _splitk_bwd_call(
        H,
        k_splits=2,
        block_M=64,
        block_N=32,
        block_K=16,
        num_stages=2,
        threads=128,
        red_M=4,
    )(As, Cs, R, delta, dGates)

    ref = lstm_bwd_ref(As, Cs, R, delta)
    assert_bwd_close(dGates, ref)
