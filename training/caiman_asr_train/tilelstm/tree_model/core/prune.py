import warnings
from collections.abc import Callable
from functools import cache

import joblib
import numpy as np
import torch
import xgboost
from beartype import beartype

from caiman_asr_train.autotune.bench import tl_log
from caiman_asr_train.autotune.cache_io import cache_read
from caiman_asr_train.tilelstm.autotune.kernel import Kernel
from caiman_asr_train.tilelstm.tree_model.core.features import featurize
from caiman_asr_train.tilelstm.tree_model.core.fingerprint import (
    BUNDLE_REL,
    source_data_fingerprint,
)
from caiman_asr_train.tilelstm.tree_model.core.specs import GpuSpec, host_spec

_TRAIN_CMD = "python -m caiman_asr_train.tilelstm.tree_model.core.train"


@beartype
def _bundle_is_current(bundle: dict) -> bool:
    """
    True if the bundle's stored ``fingerprint`` matches ``source_data_fingerprint()``. A
    mismatch means featurization, training code, or data changed, so ``ensure_bundle``
    rebuilds rather than risk ranking on a mismatched feature vector.
    """
    return bundle.get("fingerprint") == source_data_fingerprint()


@cache
@beartype
def ensure_bundle() -> dict:
    """
    Load the cached bundle, (re)training it from the packed ``data/*.jsonl.gz`` if it is
    absent or stale (then caching it), so the tree model is always available and current.
    Retraining needs CUDA (as does the autotune path that gets here). ``@cache``d: the
    load-or-train runs once per process; the returned dict is read-only, so sharing the
    instance is safe.
    """
    # Deferred: importing tilelstm reaches this module via the LSTM dispatch, and a
    # module-scope import of the trainer would put it in sys.modules before
    # `python -m ...core.train` executes it (runpy's double-import warning).
    from caiman_asr_train.tilelstm.tree_model.core.train import build_bundle

    bundle = load_bundle()
    if bundle is not None and _bundle_is_current(bundle):
        return bundle

    if bundle is None:
        reason = "no tree-model bundle cached yet"
    else:
        reason = (
            f"cached tree-model bundle is stale (source/data fingerprint "
            f"{(bundle.get('fingerprint') or '?')[:12]} != current "
            f"{source_data_fingerprint()[:12]})"
        )
    warnings.warn(
        f"{reason}; (re)training it from the packed data (one-off, then cached). "
        f"Run `{_TRAIN_CMD}` to rebuild it explicitly.",
        stacklevel=2,
    )
    return build_bundle()  # builds and caches the bundle


@beartype
def load_bundle() -> dict | None:
    """
    The trained model bundle from the cache, or ``None`` if it has not been trained yet
    (a cache miss, or the cache is disabled). Low-level loader; callers want
    ``ensure_bundle``, which trains-on-miss and rebuilds a stale bundle. Warns (but does not
    fail) if the bundle's UBJ boosters were built under a different xgboost version than is
    installed -- UBJ is version-stable, so this is advisory.
    """
    bundle = cache_read(BUNDLE_REL, joblib.load)
    if bundle is None:
        return None

    built_with = bundle.get("xgboost_version")
    installed = xgboost.__version__
    if built_with is not None and built_with != installed:
        warnings.warn(
            f"model bundle was built with xgboost {built_with} but {installed} is "
            f"installed; UBJ boosters are version-stable, but retrain with "
            f"`{_TRAIN_CMD}` if predictions look wrong",
            stacklevel=2,
        )
    return bundle


@cache
@beartype
def _load_booster(raw: bytes) -> xgboost.core.Booster:
    """
    Deserialize a UBJ booster blob into a ``Booster``. ``@cache``d on the bytes so each
    backend's booster is loaded once per process, not once per ranked shape. It loads
    CPU-side, so ranking never touches CUDA.
    """
    booster = xgboost.core.Booster()
    booster.load_model(bytearray(raw))
    return booster


@beartype
def rank_configs(
    bundle: dict,
    spec: GpuSpec,
    configs: list[dict],
    backend: Kernel,
    H: int,
    tune_batch: int,
) -> list[dict]:
    """
    Sort ``configs`` ascending by the tree's predicted log-slowdown (fastest first).

    Pure ranking helper, independent of K so it is unit-testable with a stub booster. The
    target was within-shape ``log(latency) - log(shape_best)``, so a lower prediction means
    predicted-faster. The bundle must contain a booster for every backend (enforced at build
    time), so a missing one is asserted rather than absorbed.
    """
    boosters = bundle["boosters"]
    if not configs:
        return configs
    assert backend in boosters, (
        f"tree bundle has no booster for backend {backend!r}; it must cover all "
        f"backends -- rebuild with `{_TRAIN_CMD}`"
    )

    X = featurize(configs, backend, H, tune_batch, spec)
    booster = _load_booster(boosters[backend])
    # inplace_predict wants a contiguous float32 matrix and returns float32; argsort on
    # it is exact (rank-only, so the constant base_score is irrelevant).
    preds = booster.inplace_predict(np.ascontiguousarray(X, dtype=np.float32))
    order = np.argsort(preds, kind="stable")
    return [configs[i] for i in order]


# Relative gap above which a measured host peak is too far from its recorded reference to
# be normal clock/thermal/measurement scatter.
_SPEC_DRIFT_TOL = 0.25


@beartype
def _warn_on_spec_drift(measured: GpuSpec, recorded: GpuSpec) -> None:
    """
    Warn if any measured host peak (``gemm_flops`` / ``cuda_flops`` / ``mem_bw``) differs
    from the trained reference by more than ``_SPEC_DRIFT_TOL`` (relative). Ranking still
    uses the trained spec, so a large gap means it no longer describes this host.
    """
    drifted = []
    for field in ("gemm_flops", "cuda_flops", "mem_bw"):
        ref, got = getattr(recorded, field), getattr(measured, field)
        if ref > 0 and abs(got - ref) / ref > _SPEC_DRIFT_TOL:
            drifted.append(f"{field} measured {got:.3g} vs recorded {ref:.3g}")
    if drifted:
        warnings.warn(
            f"running GPU {measured.gpu!r} measures far from its trained spec "
            f"({'; '.join(drifted)}); ranking still uses the spec the model was "
            f"trained on -- re-sweep this GPU (gen_data) + retrain if the hardware "
            f"changed",
            stacklevel=3,
        )


@cache
@beartype
def make_tree_ranker(
    k: int = 50,
    k_splitk: int = 75,
    spec: GpuSpec | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> Callable:
    """
    Return a ranker ``(configs, backend, H, tune_batch) -> configs`` for the autotuner,
    keeping the top predicted-fastest configs per shape (``k_splitk`` for split-K, whose
    config space is ~10x larger, ``k`` otherwise). Pass it as the ``config_ranker=`` kwarg
    to ``tuners.tune``.

    ``@cache``d on the arguments so ``tuners.tune``'s memo (which keys on the ranker, and
    closures hash by identity) stays hot across raw batch sizes that share a tune bucket.

    The GPU spec is resolved once (when ``spec`` is None): measure the device ``device``
    names, then featurize on the bundle's ``trained_specs`` entry for it (matching train
    time), warning on drift. An untrained GPU falls back to the measured spec with a
    warning. Pass an explicit ``spec`` to rank for a different device or avoid CUDA.
    """
    bundle = ensure_bundle()
    if spec is None:
        # Measure the GPU the caller is selecting for: the index device carries, else the
        # current device (respects torch.cuda.set_device / a DDP rank), never a blind 0.
        if device is not None and device.index is not None:
            index = device.index
        else:
            index = torch.cuda.current_device()
        measured = host_spec(index)
        trained_specs = bundle.get("trained_specs") or {}
        recorded = trained_specs.get(measured.gpu)
        if recorded is None:
            warnings.warn(
                f"the running GPU {measured.gpu!r} is not in the tree model's training set "
                f"{sorted(trained_specs)}; ranking with its measured throughput peaks, but "
                f"the prediction is an extrapolation beyond the trained GPUs",
                stacklevel=2,
            )
            spec = measured
        else:
            # Featurize on the spec the model trained this GPU on; flag a drifted host.
            recorded = GpuSpec(**recorded)
            _warn_on_spec_drift(measured, recorded)
            spec = recorded

    # The H range the model saw; outside it the tree extrapolates and may mis-prune.
    trained_h = bundle.get("trained_h") or []
    h_lo, h_hi = (min(trained_h), max(trained_h)) if trained_h else (None, None)
    warned_h: set[int] = set()

    def ranker(configs, backend, H, tune_batch):
        if h_lo is not None and not (h_lo <= H <= h_hi) and H not in warned_h:
            warned_h.add(H)
            warnings.warn(
                f"H={H} is outside the tree model's trained H range [{h_lo}, {h_hi}]; "
                f"ranking is an extrapolation and may mis-prune the top-K (a slower kernel "
                f"could be selected) -- add an H={H} sweep and retrain to cover it",
                stacklevel=2,
            )
        top_k = k_splitk if backend.is_splitk() else k
        top = rank_configs(bundle, spec, configs, backend, H, tune_batch)[:top_k]
        if verbose:
            tl_log(
                "tree-topk",
                f"{backend} H={H} B={tune_batch}: "
                f"keeping {len(top)}/{len(configs)} configs ({spec.gpu})",
            )
        return top

    return ranker
