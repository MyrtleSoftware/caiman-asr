"""
Feature extraction for the config-ranking tree model: read the ``full_ranked`` sweep and
map each candidate config at a ``(backend, H, B)`` shape to its fixed 21-column feature row.
"""

import gzip
import json
import math

import numpy as np
import tilelang
from beartype import beartype

from caiman_asr_train.tilelstm.autotune.configs_smem import (
    coop_bwd_smem_bytes,
    coop_fwd_smem_bytes,
    persistent_bwd_smem_bytes,
    persistent_fwd_smem_bytes,
    splitk_smem_bytes,
)
from caiman_asr_train.tilelstm.autotune.kernel import Kernel
from caiman_asr_train.tilelstm.tree_model.core.specs import GpuSpec


@beartype
def load_shapes(path: str) -> tuple[GpuSpec, list[dict]]:
    """
    The ``GpuSpec`` to featurize on and the shape records, from a gzipped full_ranked
    JSONL. Each shape: ``backend, H, tune_batch, ranked`` (ranked = list of
    {config, latency_ms}).
    """
    header_spec, shapes = None, []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("record") == "header":
                if r.get("spec") is not None:
                    header_spec = GpuSpec(**r["spec"])
            elif r.get("record") == "shape":
                shapes.append(r)
    if header_spec is None:
        raise RuntimeError(
            f"{path}: no header with an embedded 'spec'; regenerate the sweep with "
            f"gen_data (which embeds the measured host spec)"
        )
    return header_spec, shapes


@beartype
def _knob_col(configs: list[dict], key: str) -> np.ndarray:
    """
    Backend-specific knob column (0.0 where absent; bools -> 0.0/1.0).
    """

    def val(cfg: dict) -> float:
        v = cfg.get(key, 0)
        if isinstance(v, bool):
            return float(v)
        return float(v) if v is not None else 0.0

    return np.array([val(c) for c in configs], dtype=np.float64)


@beartype
def _smem_bytes(cfg: dict, backend: Kernel) -> int:
    """
    Authoritative per-block shared-memory bytes for ``cfg`` (autotune.configs_smem).
    """
    if backend.is_splitk():
        return splitk_smem_bytes(cfg)
    if backend.is_forward():
        if backend.is_persistent():
            return persistent_fwd_smem_bytes(cfg)
        return coop_fwd_smem_bytes(cfg)
    if backend.is_persistent():
        return persistent_bwd_smem_bytes(cfg)
    return coop_bwd_smem_bytes(cfg)


@beartype
def featurize(
    configs: list[dict], backend: Kernel, H: int, B: int, spec: GpuSpec
) -> np.ndarray:
    """
    The ``(len(configs), 21)`` float64 feature matrix for ``configs`` at one
    ``(backend, H, B)`` shape -- one row per config, column order defined by ``cols``
    below (the pin test in test_tree_model locks the mapping). The four derived geometry
    features are evaluated at ``spec`` for cross-arch generalisation.
    """
    n = len(configs)
    bM = np.array([c["block_M"] for c in configs], dtype=np.int64)
    bN = np.array([c["block_N"] for c in configs], dtype=np.int64)

    m_tiles, n_tiles = tilelang.cdiv(B, bM), tilelang.cdiv(H, bN)
    ks = np.array([c.get("k_splits", 1) or 1 for c in configs], dtype=np.int64)
    grid = n_tiles * ks if backend.is_splitk() else m_tiles * n_tiles
    bps = np.array([c.get("blocks_per_sm", 1) or 1 for c in configs], dtype=np.int64)
    lanes = bps * spec.sms if backend.is_persistent() else spec.sms

    padded_flops = (m_tiles * bM) * (n_tiles * bN) * (4 * H * 2)
    wave_quant = tilelang.cdiv(grid, lanes) * lanes / grid
    smem_bytes = np.array([_smem_bytes(c, backend) for c in configs], dtype=np.float64)

    # Coarse cross-arch traffic proxy: both GEMM operands re-streamed by the tile count on
    # the other axis; the contraction width is H forward, 4H backward/split-K.
    act_width = H if backend.is_forward() else 4 * H
    reread_bytes = n_tiles * (B * act_width * 2) + m_tiles * (4 * H * H * 2)
    # split-K also writes + re-reads k_splits fp32 partial [B,H] buffers for its reduction.
    if backend.is_splitk():
        reread_bytes = reread_bytes + ks * (2 * B * H * 4)

    ones = np.ones(n, dtype=np.float64)
    # 21-column feature set; the bundle fingerprint pins this exact order.
    cols = [
        bM.astype(np.float64),
        bN.astype(np.float64),
        np.array([c["block_K"] for c in configs], dtype=np.float64),
        np.array([c["num_stages"] for c in configs], dtype=np.float64),
        np.array([c["threads"] for c in configs], dtype=np.float64),
        _knob_col(configs, "k_splits"),
        _knob_col(configs, "res_kc"),
        _knob_col(configs, "red_M"),
        _knob_col(configs, "blocks_per_sm"),
        _knob_col(configs, "policy"),
        _knob_col(configs, "wpf"),
        float(H) * ones,
        float(B) * ones,
        math.log2(B) * ones,
        spec.gemm_flops * ones,
        spec.l2_bytes * ones,
        float(spec.arch_sm) * ones,
        padded_flops.astype(np.float64),
        wave_quant,
        smem_bytes,
        reread_bytes.astype(np.float64),
    ]
    return np.column_stack(cols)
