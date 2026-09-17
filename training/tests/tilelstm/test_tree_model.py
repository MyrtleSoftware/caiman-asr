"""
Tests for the tree model's data loading and featurization: the ``load_shapes`` spec
contract and the ``featurize`` matrix layout/formula pins. CUDA-free.
"""

import gzip
import json
import math
import os
from dataclasses import asdict

import numpy as np
import pytest

from caiman_asr_train.tilelstm.autotune.kernel import Kernel
from caiman_asr_train.tilelstm.tree_model.core.features import featurize, load_shapes
from caiman_asr_train.tilelstm.tree_model.core.specs import GpuSpec

# Golden copy of featurize's column order, pinned here (not imported) so it locks the
# real layout rather than trusting it.
_COLUMNS = (
    "block_M block_N block_K num_stages threads k_splits res_kc red_M "
    "blocks_per_sm policy wpf H B log2_B gemm_flops l2_bytes arch_sm "
    "padded_flops wave_quant smem_bytes reread_bytes"
).split()

_MEASURED = GpuSpec(
    gpu="NVIDIA H200 Imaginary Edition",
    arch_sm=90,
    gemm_flops=900e12,
    cuda_flops=60e12,
    mem_bw=4000e9,
    sms=132,
    l2_bytes=50.0 * 1024 * 1024,
)

# 4090 reference scalars (host_spec measured), pinned so the formula test has stable,
# hand-checkable inputs.
_SPEC_4090 = GpuSpec(
    gpu="NVIDIA GeForce RTX 4090",
    arch_sm=89,
    gemm_flops=155.2e12,
    cuda_flops=53.3e12,
    mem_bw=911e9,
    sms=128,
    l2_bytes=72.0 * 1024 * 1024,
)


def _write_sweep(path, device, spec=None):
    """
    A minimal full_ranked JSONL: one header (optionally with an embedded spec) and one
    two-config fwd_c shape record (complete configs, so ``build_shapes`` can featurize).
    """
    header = {"record": "header", "device": device, "kind": "full_ranked"}
    if spec is not None:
        header["spec"] = asdict(spec)
    cfg = {
        "block_M": 64,
        "block_N": 32,
        "block_K": 64,
        "num_stages": 2,
        "threads": 128,
        "dtype_lo": "float16",
        "dtype_hi": "float32",
    }
    shape = {
        "record": "shape",
        "backend": "fwd_c",
        "H": 1024,
        "tune_batch": 64,
        "ranked": [
            {"config": cfg, "latency_ms": 0.1},
            {"config": {**cfg, "block_M": 32}, "latency_ms": 0.2},
        ],
    }
    with gzip.open(path, "wt") as fh:
        fh.write(json.dumps(header) + "\n")
        fh.write(json.dumps(shape) + "\n")


def test_read_data_dir_prefers_populated_external(tmp_path, monkeypatch):
    """
    Reads resolve to the out-of-tree datasets location only once it actually holds a
    training sweep; otherwise the in-tree ``data/`` development fallback wins. A stray
    non-sweep file (e.g. a custom_lstm baseline) must not flip the resolution.
    """
    import caiman_asr_train.tilelstm.tree_model as tree_model

    monkeypatch.setattr(tree_model, "EXTERNAL_DATA_DIR", str(tmp_path / "absent"))
    assert tree_model.read_data_dir() == tree_model.DATA_DIR

    external = tmp_path / "tile_tree_data"
    external.mkdir()
    monkeypatch.setattr(tree_model, "EXTERNAL_DATA_DIR", str(external))
    assert tree_model.read_data_dir() == tree_model.DATA_DIR  # exists but empty

    (external / "custom_lstm_h200.jsonl.gz").write_bytes(b"")
    assert tree_model.read_data_dir() == tree_model.DATA_DIR  # baseline is not a sweep

    (external / "full_ranked_h200.jsonl.gz").write_bytes(b"")
    assert tree_model.read_data_dir() == str(external)


def test_checked_in_ci_sweep_covers_every_backend(tmp_path, monkeypatch):
    """
    A cold CI checkout has no datasets mount or cached model bundle, so its in-tree
    fallback must contain enough real sweep data to build every backend's booster.
    """
    pytest.importorskip("xgboost")  # train.py imports xgboost at module scope
    import caiman_asr_train.tilelstm.tree_model as tree_model
    from caiman_asr_train.tilelstm.tree_model.core.train import build_shapes, data_files

    monkeypatch.setattr(tree_model, "EXTERNAL_DATA_DIR", str(tmp_path / "absent"))
    fixture = os.path.join(tree_model.DATA_DIR, "full_ranked_ci.jsonl.gz")
    assert fixture in data_files()

    per_backend, _specs = build_shapes([fixture])
    assert set(per_backend) == set(tree_model.BACKENDS)
    assert all(len(shapes) == 1 for shapes in per_backend.values())


def test_load_shapes_uses_embedded_spec(tmp_path):
    """
    Every sweep is self-describing: the header-embedded spec is what featurization gets,
    with no spec table anywhere to edit.
    """
    path = str(tmp_path / "full_ranked_h200.jsonl.gz")
    _write_sweep(path, _MEASURED.gpu, spec=_MEASURED)
    spec, shapes = load_shapes(path)
    assert spec == _MEASURED
    assert len(shapes) == 1


def test_load_shapes_without_embedded_spec_raises(tmp_path):
    """
    A file predating the embedded spec fails loudly (pointing at gen_data) rather than
    training on a guessed spec.
    """
    path = str(tmp_path / "full_ranked_h200.jsonl.gz")
    _write_sweep(path, _MEASURED.gpu, spec=None)
    with pytest.raises(RuntimeError, match="embedded 'spec'"):
        load_shapes(path)


def test_build_shapes_warns_on_duplicate_cells(tmp_path):
    """
    The same ``(gpu, backend, H, B)`` cell in two files (a tagged subset sweep left
    beside a full sweep of the same GPU) is kept but double-weights the fit, so
    ``build_shapes`` must warn about it.
    """
    pytest.importorskip("xgboost")  # train.py imports xgboost at module scope
    from caiman_asr_train.tilelstm.tree_model.core.train import build_shapes

    full = str(tmp_path / "full_ranked_h200.jsonl.gz")
    tagged = str(tmp_path / "full_ranked_h200_fwd_c.jsonl.gz")
    _write_sweep(full, _MEASURED.gpu, spec=_MEASURED)
    _write_sweep(tagged, _MEASURED.gpu, spec=_MEASURED)

    with pytest.warns(UserWarning, match="duplicate"):
        per_backend, _specs = build_shapes([full, tagged])
    assert len(per_backend[Kernel.fwd_c]) == 2  # both kept, only warned

    # Distinct files without overlap stay silent.
    per_backend, _specs = build_shapes([full])
    assert len(per_backend[Kernel.fwd_c]) == 1


def test_featurize_matrix_pins_formulas():
    """
    ``featurize`` returns an ``(n, 21)`` float64 matrix in ``_COLUMNS`` order; pin one
    hand-computed row (fwd_c geometry + coop-forward SMEM at H=1024, B=48 on the 4090
    reference scalars) so a refactor can't silently move a column or change a formula.
    """
    cfg = {
        "block_M": 64,
        "block_N": 32,
        "block_K": 64,
        "num_stages": 2,
        "threads": 128,
        "dtype_lo": "float16",
        "dtype_hi": "float32",
    }
    spec = _SPEC_4090  # sms=128
    X = featurize([cfg, {**cfg, "block_M": 32}], Kernel.fwd_c, 1024, 48, spec)
    assert X.shape == (2, len(_COLUMNS)) and X.dtype == np.float64

    row = dict(zip(_COLUMNS, X[0]))
    assert row["block_M"] == 64.0 and row["block_N"] == 32.0
    assert row["k_splits"] == 0.0 and row["wpf"] == 0.0  # absent knobs -> 0.0
    assert row["H"] == 1024.0 and row["B"] == 48.0
    assert row["log2_B"] == math.log2(48)
    assert row["gemm_flops"] == spec.gemm_flops and row["arch_sm"] == 89.0
    # m_tiles=1, n_tiles=32: padded = (1*64)*(32*32)*(4*1024*2); grid=32 on 128 SMs.
    assert row["padded_flops"] == 536870912.0
    assert row["wave_quant"] == 4.0
    # coop fwd matmul phase: 2 stages * (64*64 + 4*32*64) * fp16 > epilogue 16384.
    assert row["smem_bytes"] == 49152.0
    # 32*(48*1024*2) activation re-stream + 1*(4*1024*1024*2) weight stream.
    assert row["reread_bytes"] == 11534336.0

    assert X[1, 0] == 32.0  # the second config's block_M lands in the same column
    assert featurize([], Kernel.fwd_c, 1024, 48, spec).shape == (0, 21)


# --------------------------------------------------------------------------------------
# xgboost bundle round-trip (CPU-only; skipped when xgboost is not installed). Kept out of
# module scope so the featurize pins above collect even without xgboost.
# --------------------------------------------------------------------------------------


def _fwd_c_configs(block_ms):
    """
    A list of fwd_c config dicts identical but for ``block_M`` (the only varied knob).
    """
    base = {
        "block_N": 32,
        "block_K": 64,
        "num_stages": 2,
        "threads": 128,
        "dtype_lo": "float16",
        "dtype_hi": "float32",
    }
    return [{**base, "block_M": bm} for bm in block_ms]


def _tiny_model(xgb, configs, spec):
    """
    A 4-tree CPU xgboost regressor fit on the real featurize matrix (target = block_M).
    """
    X = np.ascontiguousarray(
        featurize(configs, Kernel.fwd_c, 1024, 48, spec), dtype=np.float32
    )
    y = np.array([c["block_M"] for c in configs], dtype=np.float32)
    model = xgb.XGBRegressor(
        n_estimators=4, max_depth=2, tree_method="hist", device="cpu", random_state=0
    )
    model.fit(X, y)
    return model


def test_prune_rank_configs_ubj_roundtrip():
    """
    A UBJ booster in a bundle dict ranks configs: ``rank_configs`` returns a permutation of
    the same config objects, ordered exactly by the booster's own predictions. CUDA-free.
    """
    xgb = pytest.importorskip("xgboost")
    from caiman_asr_train.tilelstm.tree_model.core import prune

    spec = _SPEC_4090
    model = _tiny_model(xgb, _fwd_c_configs([32, 48, 64, 96, 128] * 8), spec)
    bundle = {
        "fingerprint": "test",
        "booster_kind": "xgboost",
        "xgboost_version": xgb.__version__,
        "trained_h": [1024],
        "trained_specs": {},
        "boosters": {Kernel.fwd_c: bytes(model.get_booster().save_raw("ubj"))},
    }

    configs = _fwd_c_configs([64, 32, 128, 48, 96])
    ranked = prune.rank_configs(bundle, spec, configs, Kernel.fwd_c, 1024, 48)
    assert len(ranked) == len(configs)
    assert {id(c) for c in ranked} == {id(c) for c in configs}  # a pure permutation

    booster = prune._load_booster(bundle["boosters"][Kernel.fwd_c])
    X = np.ascontiguousarray(
        featurize(configs, Kernel.fwd_c, 1024, 48, spec), dtype=np.float32
    )
    order = np.argsort(booster.inplace_predict(X), kind="stable")
    assert [id(c) for c in ranked] == [id(configs[i]) for i in order]

    assert prune.rank_configs(bundle, spec, [], Kernel.fwd_c, 1024, 48) == []


def test_build_bundle_field_contract(monkeypatch):
    """
    ``build_bundle`` assembles the deployable dict with the xgboost bundle contract keys
    (``booster_kind``/``xgboost_version``/``fingerprint``) + raw UBJ bytes per backend.
    """
    xgb = pytest.importorskip("xgboost")
    from caiman_asr_train.tilelstm.tree_model.core import train

    # Disable the cache so build_bundle assembles the dict without touching ~/.tilelang.
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    model = _tiny_model(xgb, _fwd_c_configs([32, 48, 64, 96] * 6), _SPEC_4090)
    trained = train.TrainedModels(
        models={Kernel.fwd_c: model},
        trained_h=[1024],
        trained_specs={},
        per_backend={},
    )
    bundle = train.build_bundle(trained)
    for key in (
        "fingerprint",
        "booster_kind",
        "xgboost_version",
        "trained_h",
        "trained_specs",
        "boosters",
    ):
        assert key in bundle
    assert bundle["booster_kind"] == "xgboost"
    assert bundle["xgboost_version"] == xgb.__version__
    assert isinstance(bundle["boosters"][Kernel.fwd_c], bytes)
