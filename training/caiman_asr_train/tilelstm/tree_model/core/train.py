"""
Train the per-backend tree models and cache the bundle ``prune.py`` loads to rank autotune
configs: one gradient-boosted regressor per backend (all sharing the one ``PARAMS``
hyperparameter config), fit on every ``data/full_ranked_*.jsonl.gz`` present, with target =
within-shape ``log(latency) - log(shape_best)``.

Run in the container (needs a GPU -- ``new_model`` trains with ``device="cuda"``)::

    python -m caiman_asr_train.tilelstm.tree_model.core.train
"""

import glob
import os
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import xgboost
from beartype import beartype

from caiman_asr_train.autotune.cache_io import cache_write
from caiman_asr_train.tilelstm.autotune.kernel import Kernel
from caiman_asr_train.tilelstm.tree_model import (
    BACKENDS,
    TRAIN_DATA_GLOB,
    read_data_dir,
)
from caiman_asr_train.tilelstm.tree_model.core.features import featurize, load_shapes
from caiman_asr_train.tilelstm.tree_model.core.fingerprint import (
    BUNDLE_REL,
    source_data_fingerprint,
)
from caiman_asr_train.tilelstm.tree_model.core.metrics import excess_stats, k_metric
from caiman_asr_train.tilelstm.tree_model.core.specs import GpuSpec


@beartype
@dataclass(frozen=True)
class Shape:
    """
    One ``(gpu, backend, H, B)`` cell: its featurized configs ``X`` and measured latencies
    ``lat`` (``best`` = the shape's fastest, the log-slowdown baseline).
    """

    gpu: str
    backend: Kernel
    H: int
    B: int
    X: np.ndarray
    lat: np.ndarray

    @property
    def best(self) -> float:
        """
        The shape's fastest measured latency -- the per-shape log-slowdown baseline.
        """
        return float(self.lat.min())


# backend -> its shapes (build_shapes result).
PerBackend = dict[Kernel, list[Shape]]


@beartype
def data_files() -> list[str]:
    """
    Every ``full_ranked_*.jsonl.gz`` in the resolved read dir (``read_data_dir``: the
    out-of-tree datasets location when populated, else the in-tree ``data/``), so a
    dropped-in new GPU sweep is found automatically.
    """
    data_dir = read_data_dir()
    found = sorted(glob.glob(os.path.join(data_dir, TRAIN_DATA_GLOB)))
    if not found:
        # RuntimeError, not SystemExit: reachable from ensure_bundle's train-on-miss
        # inside production autotuning, where a BaseException would kill the process.
        raise RuntimeError(
            f"no training data in {data_dir} (run gen_data.py to create it)"
        )
    return found


# Shared xgboost kwargs from the Phase-H selection (tree_model_scratch/results/
# shared_selection.json): ONE config for all backends (still one separately-fitted booster
# per backend), chosen as minimal size at continuous-metric baseline parity across all
# backends -- the only evaluated shared config whose worst-seed K tails stay inside the
# production margin on every backend, split-K included. ``device`` and ``random_state`` are
# fixed in new_model. This dict is hashed by the fingerprint, so editing it rebuilds the
# bundle.
PARAMS: dict = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "max_depth": 0,
    "max_leaves": 31,
    "min_child_weight": 19.165,
    "gamma": 0.0,
    "reg_lambda": 0.294,
    "reg_alpha": 0.314083,
    "learning_rate": 0.11119,
    "subsample": 0.8545,
    "colsample_bytree": 1.0,
    "max_bin": 128,
    "n_estimators": 384,
}


@beartype
def new_model() -> xgboost.XGBRegressor:
    """
    A fresh xgboost regressor from the shared PARAMS config, with a fixed seed for
    reproducible rebuilds. Training requires CUDA (like ``gen_data``).
    """
    return xgboost.XGBRegressor(**PARAMS, device="cuda", random_state=0)


@beartype
def fit_on(train_shapes: list[Shape]) -> xgboost.XGBRegressor:
    """
    Fit one regressor on the stacked rows of ``train_shapes`` (target = log-slowdown). X is
    cast to float32 -- xgboost's native dtype, matching the ranker's inplace_predict path.
    """
    Xs, ys = [], []
    for sh in train_shapes:
        Xs.append(sh.X)
        ys.append(np.log(sh.lat) - np.log(sh.best))
    m = new_model()
    m.fit(np.concatenate(Xs).astype(np.float32), np.concatenate(ys))
    return m


@beartype
def build_shapes(paths: list[str]) -> tuple[PerBackend, dict[str, GpuSpec]]:
    """
    ``({backend: [Shape, ...]}, {gpu name: its spec})`` across the given data files (call
    ``data_files()`` for the committed set). Each file's spec comes from its embedded
    header; the spec map is what the bundle records as ``trained_specs``, so the ranker
    can featurize a trained GPU exactly as training did. Shapes with < 2 configs (nothing
    to rank) are skipped.

    Warns when the same ``(gpu, backend, H, B)`` cell appears more than once across the
    files -- e.g. a tagged ``--backends``-subset sweep left beside a full sweep of the
    same GPU. Both copies are kept (behaviour unchanged), but the duplicated shapes are
    double-weighted in the fit, so the stray file should be removed.
    """
    per_backend: PerBackend = defaultdict(list)
    specs: dict[str, GpuSpec] = {}
    seen: set[tuple] = set()
    dups: list[tuple] = []
    for path in paths:
        spec, shapes = load_shapes(path)
        specs[spec.gpu] = spec
        for sh in shapes:
            bk = Kernel(sh["backend"])
            H, B = sh["H"], sh["tune_batch"]
            ranked = sh["ranked"]
            if len(ranked) < 2:
                continue
            cell = (spec.gpu, bk, H, B)
            if cell in seen:
                dups.append(cell)
            seen.add(cell)
            X = featurize([e["config"] for e in ranked], bk, H, B, spec)
            lat = np.array([e["latency_ms"] for e in ranked], dtype=np.float64)
            per_backend[bk].append(Shape(spec.gpu, bk, H, B, X, lat))
    if dups:
        warnings.warn(
            f"{len(dups)} duplicate (gpu, backend, H, B) cells across the data files "
            f"(e.g. {dups[0]}); a tagged subset sweep probably coexists with a full "
            f"sweep of the same GPU -- remove one, or the duplicated shapes are "
            f"double-weighted in training",
            stacklevel=2,
        )
    return per_backend, specs


@beartype
@dataclass(frozen=True)
class TrainedModels:
    """
    The per-backend fitted models plus what the bundle records: the H values covered
    (``prune`` warns outside this range) and the per-GPU sweep specs. ``per_backend`` holds
    the featurized shapes the fit consumed, so the CLI's in-sample readout reuses them; it
    never enters the bundle.
    """

    models: dict[Kernel, xgboost.XGBRegressor]
    trained_h: list[int]
    trained_specs: dict[str, GpuSpec]
    per_backend: PerBackend


@beartype
def train() -> TrainedModels:
    """
    Fit one model per backend on all committed data.

    Every backend must get a model: the ranker asserts the bundle covers all backends, so
    an incomplete sweep is a hard error here rather than a bundle that silently can't rank.
    """
    pb, trained_specs = build_shapes(data_files())
    models = {}
    for bk in BACKENDS:
        shapes = pb.get(bk)
        if shapes:
            models[bk] = fit_on(shapes)
    missing = [bk for bk in BACKENDS if bk not in models]
    if missing:
        raise RuntimeError(
            f"no training data for backend(s) {missing} under {read_data_dir()}; the "
            f"bundle must cover all {len(BACKENDS)} backends -- add full_ranked sweeps "
            f"(gen_data.py) that exercise them, or drop them from BACKENDS"
        )
    trained_h = sorted({s.H for shapes in pb.values() for s in shapes})
    return TrainedModels(
        models=models,
        trained_h=trained_h,
        trained_specs=trained_specs,
        per_backend=pb,
    )


@beartype
def build_bundle(trained: TrainedModels | None = None) -> dict:
    """
    Assemble and cache the deployable bundle from ``trained`` (fitting via ``train()`` when
    not given) and return it. The joblib dict holds ``fingerprint``, ``booster_kind``,
    ``xgboost_version``, ``trained_h``, ``trained_specs`` (per-GPU sweep specs), and
    ``boosters`` (one per backend). Shared by the CLI and ``prune.make_tree_ranker``'s
    train-if-missing path.
    """
    if trained is None:
        trained = train()

    bundle = {
        "fingerprint": source_data_fingerprint(),
        "booster_kind": "xgboost",
        "xgboost_version": xgboost.__version__,
        "trained_h": trained.trained_h,
        # Plain dicts, not GpuSpec instances, so a GpuSpec field change can't break an old
        # bundle (the fingerprint rebuilds it instead).
        "trained_specs": {
            gpu: asdict(spec) for gpu, spec in trained.trained_specs.items()
        },
        # Raw UBJ booster bytes, not pickled estimators: UBJ is version-stable, so an
        # xgboost upgrade can't break deserialization the way a cross-version pickle can.
        "boosters": {
            bk: bytes(model.get_booster().save_raw("ubj"))
            for bk, model in trained.models.items()
        },
    }

    cache_write(BUNDLE_REL, lambda tmp: joblib.dump(bundle, tmp, compress=3))

    return bundle


@beartype
def insample_ks(
    models: dict[Kernel, xgboost.XGBRegressor],
    per_backend: PerBackend,
    tol: float,
) -> dict[Kernel, list[float]]:
    """
    Per-backend in-sample K: rank each shape's own configs with the model trained on them
    and take ``k_metric`` at ``tol``. One batched ``predict`` per backend, sliced back per
    shape (a per-shape predict pays the estimator's per-call overhead thousands of times). A
    quick post-train sanity readout; optimistic, since there is no hold-out.
    """
    out: dict[Kernel, list[float]] = {}
    for bk, model in models.items():
        shapes = per_backend.get(bk, [])
        if not shapes:
            out[bk] = []
            continue
        # The booster trained on cuda but the rows are cpu numpy; align devices so the
        # predict doesn't warn and fall back to DMatrix (float32 = the ranker's path).
        # On a COPY: this readout must not mutate the fitted model a caller may still
        # serialize.
        bst = model.get_booster().copy()
        bst.set_param({"device": "cpu"})
        preds = bst.inplace_predict(
            np.concatenate([sh.X for sh in shapes]).astype(np.float32)
        )
        bounds = np.cumsum([len(sh.X) for sh in shapes])[:-1]
        out[bk] = [
            k_metric(p, sh.lat, sh.best, tol)
            for p, sh in zip(np.split(preds, bounds), shapes)
        ]
    return out


@beartype
def print_insample_report(
    models: dict[Kernel, xgboost.XGBRegressor],
    per_backend: PerBackend,
    pct: float = 3.0,
) -> None:
    """
    Print the in-sample within-``pct``% K :class:`ExcessStats` per backend (+ overall grand
    total) for the freshly fitted ``models``, with each backend's tree count -- a quick
    sanity readout of how tightly the models rank their own training data. Optimistic:
    in-sample, no hold-out.
    """
    ks_by_bk = insample_ks(models, per_backend, 1.0 + pct / 100.0)
    print(f"\nin-sample within-{pct:g}% K  (optimistic: in-sample, no hold-out)")
    print(
        f"{'backend':18}{'trees':>6}{'cells':>6}{'pPerf':>7}{'medK':>7}"
        f"{'Σe²':>9}{'rms':>7}{'maxK':>6}"
    )

    def row(label: str, trees: int, ks: list[float]) -> None:
        s = excess_stats(ks)
        print(
            f"{label:18}{trees:>6}{s.n:>6}{s.p_perfect:>7.2f}{s.medK:>7.1f}"
            f"{s.sum_sq:>9.0f}{s.rms:>7.2f}{s.maxK:>6.0f}"
        )

    all_ks: list[float] = []
    total_trees = 0
    for bk in BACKENDS:
        bk_ks = ks_by_bk.get(bk, [])
        if not bk_ks:
            continue
        trees = models[bk].get_booster().num_boosted_rounds()
        total_trees += trees
        all_ks += bk_ks
        row(bk.value, trees, bk_ks)
    if all_ks:
        row("ALL", total_trees, all_ks)


@beartype
def main() -> None:
    print("training per-backend tree models ...")
    trained = train()
    bundle = build_bundle(trained)
    print(
        f"trained {len(bundle['boosters'])} backends (H={bundle['trained_h']}); "
        f"xgboost {xgboost.__version__}; wrote bundle to the cache"
    )
    # In-sample K readout (CLI-only), kept out of build_bundle so the train-if-missing path
    # stays silent.
    print_insample_report(trained.models, trained.per_backend)


if __name__ == "__main__":
    main()
