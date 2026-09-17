"""
Tests for the LSTM-specific autotuner calibration:

* ``configs`` -- the LSTM shape predicates (``_tile_ok`` /
  ``_epilogue_reduction_ok``) and their composition with the generic
  ``mnk_atom_ok``, exactly as the production generators compose them; pinned to
  the exact formulas they replaced so the de-drift cannot silently flip a
  verdict. (The generic filters themselves are pinned in
  ``tests/autotune/test_feasibility.py``.)
* ``configs_smem.splitk_smem_bytes`` -- the split-K SMEM formula (single source of
  truth, shared with the candidate search via ``splitk_smem_bytes_at``).

All CUDA-free (pure functions), so it runs anywhere. The device residency
model (``resident_blocks`` / ``_regs_per_block`` / ``grid_resident``) is exercised
behaviorally through ``test_config_generators_yield_feasible_configs`` in
``test_autotune`` rather than re-derived here against fake ``DeviceLimits``.
"""

import pytest

from caiman_asr_train.autotune.feasibility import mnk_atom_ok
from caiman_asr_train.tilelstm.autotune.configs import (
    _epilogue_reduction_ok,
    _tile_ok,
)
from caiman_asr_train.tilelstm.autotune.configs_smem import splitk_smem_bytes

# ---------------------------------------------------------------------------
# configs: the LSTM shape predicates (_tile_ok / _epilogue_reduction_ok) composed
# with the generic mnk_atom_ok, as the production generators compose them.
# ---------------------------------------------------------------------------


# Exercises each shape filter at H=1024: the m16n8k16 WMMA-atom multiples (bM%16, bN%8,
# bK%16), block_K | H (the bK=96 tile fails it), block_N <= H, and -- the ragged-N change
# -- that a WMMA-n-multiple block_N which does NOT divide H is still feasible. (threads is
# NOT one of these filters -- it is degenerate with wmma_ok, which every generator also
# applies; the threads=2048 tile still passes these two.)
_PRED_TILES = [
    {"block_M": 64, "block_N": 16, "block_K": 64, "threads": 128},  # clean
    {"block_M": 128, "block_N": 32, "block_K": 128, "threads": 256},  # clean, 256 thr
    {"block_M": 8, "block_N": 16, "block_K": 64, "threads": 128},  # bM not %16
    {"block_M": 64, "block_N": 12, "block_K": 64, "threads": 128},  # bN not %8
    {"block_M": 64, "block_N": 16, "block_K": 24, "threads": 128},  # bK not %16
    {"block_M": 64, "block_N": 48, "block_K": 64, "threads": 128},  # ragged-N (bN ∤ H)
    {"block_M": 96, "block_N": 24, "block_K": 96, "threads": 256},  # bK ∤ 1024
    {"block_M": 64, "block_N": 16, "block_K": 64, "threads": 2048},  # threads>bM*bN
]


@pytest.mark.parametrize("cfg", _PRED_TILES)
def test_m16n8k16_and_tile_ok(cfg):
    H = 1024
    # threads is intentionally NOT unpacked: mnk_atom_ok/_tile_ok do not gate on it.
    bM, bN, bK = cfg["block_M"], cfg["block_N"], cfg["block_K"]
    # mnk_atom_ok(mult=1) + _tile_ok = the WMMA-atom multiples (mult=1 => block_N % 8,
    # the kernel's WMMA-atom tile contract) + block_N<=H + block_K | H. H need NOT divide
    # block_N (ragged-N): a non-dividing block_N that is a WMMA-n multiple and no wider than
    # H is feasible. NB threads<=bM*bN is NOT checked here -- it is degenerate with wmma_ok
    # (the generators also apply that), so cfg with threads=2048 still passes these two.
    expected = bM % 16 == 0 and bN % 8 == 0 and bK % 16 == 0 and bN <= H and H % bK == 0
    cfgH = {**cfg, "H": H}
    assert (mnk_atom_ok(cfgH, n_block_mult=1) and _tile_ok(cfgH)) == expected


def test_epilogue_reduction_ok_enforces_per_gate_block_n():
    # Forward gate-major epilogue: block_N % 8 (same-lane 4-gate gather) is STRICTER than
    # the gemm's 4*block_N atom (block_N even), so mnk_atom_ok(mult=4) alone would admit
    # block_N=4 -- _epilogue_reduction_ok is what rejects it for forward.
    even_not8 = {"block_M": 64, "block_N": 4, "block_K": 64, "threads": 128}
    assert mnk_atom_ok(even_not8, n_block_mult=4) is True  # passes the gemm atom...
    assert (
        _epilogue_reduction_ok(even_not8) is False
    )  # ...but the per-gate gather rejects
    assert _epilogue_reduction_ok({"block_M": 64, "block_N": 8, "threads": 128}) is True
    # the row constraint (threads <= 2*block_M) still bites independently
    assert (
        _epilogue_reduction_ok({"block_M": 64, "block_N": 8, "threads": 256}) is False
    )


def test_tile_ok_rejects_oversized_block_n():
    """A column tile wider than H is degenerate (ragged-N admits non-dividing block_N,
    but not block_N > H)."""
    assert not _tile_ok(
        {"block_M": 64, "block_N": 128, "block_K": 64, "threads": 128, "H": 64}
    )
    # ... but the same WMMA-n tile is feasible once H >= block_N (ragged remainder).
    assert _tile_ok(
        {"block_M": 64, "block_N": 128, "block_K": 64, "threads": 128, "H": 192}
    )


# ---------------------------------------------------------------------------
# configs_smem.splitk_smem_bytes (delegates to splitk_smem_bytes_at, the single source of
# truth the candidate search shares)
# ---------------------------------------------------------------------------


def test_splitk_smem_full_strip_has_no_tail():
    # H=1024, ks=8, bK=64 -> kchunks = (4*1024/8)/64 = 8; res_kc=8 == full strip.
    cfg = dict(
        block_M=64,
        block_N=64,
        block_K=64,
        num_stages=2,
        res_kc=8,
        k_splits=8,
        H=1024,
        dtype_lo="float16",
    )
    lo = 2
    expect = 8 * 64 * 64 * lo + 2 * 64 * 64 * lo  # strip + dG pipe, no tail
    assert splitk_smem_bytes(cfg) == expect


def test_splitk_smem_partial_strip_adds_tail():
    cfg = dict(
        block_M=64,
        block_N=64,
        block_K=64,
        num_stages=2,
        res_kc=4,
        k_splits=8,
        H=1024,
        dtype_lo="float16",
    )
    lo = 2
    expect = 4 * 64 * 64 * lo + 2 * 64 * 64 * lo + 2 * 64 * 64 * lo  # +tail rW pipe
    assert splitk_smem_bytes(cfg) == expect
