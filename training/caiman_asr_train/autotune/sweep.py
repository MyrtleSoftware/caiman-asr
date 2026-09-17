import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import tilelang  # noqa: F401  # registers the bundled TVM: `import tvm` needs it first
import tvm
from beartype import beartype
from tilelang.jit import par_compile as _par_compile
from tqdm import tqdm

from caiman_asr_train.autotune.bench import lean_bench, tl_log
from caiman_asr_train.autotune.cache_io import (
    cache_read,
    cache_write,
)

# Parallel-compile fan-out (threads driving nvcc). Env-overridable so a
# full-sweep agent can lower it WITHOUT patching. This is a SAFETY knob, not a
# speed knob (the sweep is bench-bound, not compile-bound): at high fan-out the
# wide H=1536 bwd_s compile can OOM-deadlock, and the storm of nvcc subprocesses
# / open file handles can exhaust host resources and take the whole box offline.
# Keep it low (4-8) for the full sweep; 16 has deadlocked H=1536.
_PAR_COMPILE_WORKERS = int(os.environ.get("CAIMAN_LSTM_PAR_COMPILE_WORKERS", "16"))


# Config keys constant across a whole tuning run (the shape + precision); they are pure
# noise on every per-config log line, so ``_fmt_cfg`` drops them and shows only the tuned
# tile knobs. Formatting defaults only: keys absent from a kernel's configs are no-ops,
# unknown keys pass through unabbreviated.
_CONST_CFG_KEYS = frozenset(
    {"H", "dtype_lo", "dtype_hi", "dtype", "accum_dtype", "save_gates", "fpad"}
)

# Short aliases for the verbose tile-knob keys, to keep the per-config log line terse.
_CFG_ABBREV = {
    "num_stages": "ns",
    "block_M": "bM",
    "block_N": "bN",
    "block_K": "bK",
}


@beartype
def _fmt_cfg(cfg: dict) -> str:
    """
    Compact one-line rendering of a tuned config for the log.
    """
    return " ".join(
        f"{_CFG_ABBREV.get(k, k)}={getattr(v, 'name', v)}"
        for k, v in cfg.items()
        if k not in _CONST_CFG_KEYS
    )


@beartype
def _tir_hash(tir: object) -> str:
    """
    Structural hash of a lowered PrimFunc, used as the bench-cache key for one config.
    """
    return str(tvm.ir.structural_hash(tir, map_free_vars=True))


@beartype
def _bench_cache_dir(cache_key: tuple) -> Path:
    """
    Directory (relative to the cache root) for the bench timings at one ``cache_key``.
    """
    sha = hashlib.sha256(json.dumps(cache_key, sort_keys=True, default=str).encode())
    # Entries are keyed by structural TIR hash, so multiple kernels share the dir
    # without collision.
    return Path("caiman_autotune_bench") / sha.hexdigest()


@beartype
def _load_bench_entry(bench_dir: Path, tir_hash: str) -> dict | None:
    """
    The cached bench result for a config's TIR hash, returns:
        ``{"config", "latency_ms"}``; ``latency_ms`` = ``None`` = a cached comp/launch fail
        ``None`` if not benched (or the cache is disabled).
    """
    # cache_read opens the file (binary); json.load reads bytes directly.
    return cache_read(bench_dir / f"{tir_hash}.json", json.load)


@beartype
def _store_bench_entry(
    bench_dir: Path, tir_hash: str, cfg: dict, latency_ms: float | None
) -> None:
    """
    Persist a config's bench result under its TIR hash (one file per config). ``cfg`` is
    kept for the ``ranked`` output; ``latency_ms`` ``None`` records a deterministic
    compile/launch failure so it is not retried. A no-op if the cache is disabled.
    """
    payload = {"config": cfg, "latency_ms": latency_ms}

    def _write(tmp: str) -> None:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)

    # Process-safe atomic publish (shared with the tree-model bundle); see cache_io.
    cache_write(bench_dir / f"{tir_hash}.json", _write)


@beartype
def _bench_delta(
    *,
    factory: object,
    to_bench: list[tuple[int, dict, object, str]],
    make_thunk: Callable,
    validate: Callable,
    bench_dir: Path,
    label: str,
    H: int,
    tune_batch: int,
    n_warmup: int,
    n_repeat: int,
    timing: dict[int, float],
) -> None:
    """
    Parallel-compile + benchmark + validate the to-bench DELTA -- the already-elaborated
    configs (``(idx, cfg, tir, tir_hash)``) not yet in the bench cache -- recording each
    success's latency in ``timing`` (original-index -> ms) and persisting a per-config
    cache entry for EVERY attempt (``latency_ms = None`` for a compile/launch failure, so
    an unchanged set later re-attempts nothing). A config that fails to compile is an
    expected infeasibility (SMEM cap / residency / warp-partition). Mutates ``timing`` in
    place.
    """
    funcs = [tir for _, _, tir, _ in to_bench]

    kernels = (
        _par_compile(
            funcs,
            out_idx=factory.out_idx,
            execution_backend=factory.execution_backend,
            target=factory.target,
            target_host=factory.target_host,
            verbose=factory.verbose,
            pass_configs=factory.pass_configs,
            compile_flags=factory.compile_flags,
            num_workers=_PAR_COMPILE_WORKERS,
            ignore_error=True,
        )
        if funcs
        else []
    )

    for j, kernel in enumerate(
        tqdm(kernels, total=len(kernels), desc=f"Benchmarking {label}")
    ):
        i, cfg, _, h = to_bench[j]
        # A None kernel is an (expected) compile failure -- cache it so an
        # unchanged set does not recompile it, and move on.
        if kernel is None:
            _store_bench_entry(bench_dir, h, cfg, None)
            continue
        ms = lean_bench(
            make_thunk(kernel, cfg),
            n_warmup=n_warmup,
            n_repeat=n_repeat,
        )
        if ms is None:  # launch failure (e.g. cooperative grid too large)
            _store_bench_entry(bench_dir, h, cfg, None)
            continue

        # Validate every freshly-benched config (a numerically-wrong one crashes the
        # tune, naming itself); cached configs are trusted (verdict keyed by the TIR hash).
        try:
            validate(kernel, cfg)
        except AssertionError as e:
            raise AssertionError(
                f"[TileTune/{label}] wrong numerics at H={H} B={tune_batch}: {cfg}"
            ) from e

        timing[i] = ms
        _store_bench_entry(bench_dir, h, cfg, ms)


@beartype
def autotune_select(
    *,
    factory: object,
    configs: list[dict],
    make_thunk: Callable,
    validate: Callable,
    finalize: Callable,
    cache_key: tuple,
    label: str,
    H: int,
    tune_batch: int,
    n_warmup: int = 3,
    n_repeat: int = 10,
) -> tuple[Callable, list[dict]] | None:
    """
    Pick the fastest config for one backend and return ``(finalize(kernel, cfg), ranked)``
    -- the call-ready kernel (the winner) and ``ranked``: every feasible config with its
    measured tune-batch latency, as ``{"config", "latency_ms"}`` dicts sorted ascending
    (so ``ranked[0]`` is the winner). Returns ``None`` when NO config in the set produced a
    measured time (all infeasible -- compile/launch failures, see ``_bench_delta``), so the
    caller skips this backend.

    Args:

    * ``factory`` -- a jit decorated TileLang kernel factory
    * ``make_thunk(kernel, cfg) -> () -> None`` -- a zero-arg bench thunk on the
      tune-batch buffers (wrapping with scratch alloc / counter re-zeroing as needed).
    * ``validate(kernel, cfg) -> None`` -- run on fresh buffers + ``assert_close`` vs
      the reference; an ``AssertionError`` MUST propagate. Called on every freshly
      benchmarked config (a wrong one crashes the tune, naming it) AND on the winner;
      cached configs are trusted (the verdict is keyed by the TIR hash).
    * ``finalize(kernel, cfg)`` -- adapt the kernel to the uniform call contract.
    """
    bench_dir = _bench_cache_dir(cache_key)

    elaborated = [(cfg, factory.get_tir(**cfg)) for cfg in configs]

    # Partition into cache hits (measured timings) and the to-bench DELTA. A
    # cached FAILURE (latency None) is neither re-benched nor a winner.
    timing: dict[int, float] = {}
    to_bench: list[tuple[int, dict, object, str]] = []
    for i, (cfg, tir) in enumerate(elaborated):
        h = _tir_hash(tir)
        entry = _load_bench_entry(bench_dir, h)
        if entry is None:
            to_bench.append((i, cfg, tir, h))
        elif entry.get("latency_ms") is not None:
            timing[i] = float(entry["latency_ms"])

    if to_bench:
        tl_log(
            label,
            f"tuning H={H} B={tune_batch}: benching {len(to_bench)}/{len(configs)} "
            f"configs ({len(configs) - len(to_bench)} cached)",
        )
        _bench_delta(
            factory=factory,
            to_bench=to_bench,
            make_thunk=make_thunk,
            validate=validate,
            bench_dir=bench_dir,
            label=label,
            H=H,
            tune_batch=tune_batch,
            n_warmup=n_warmup,
            n_repeat=n_repeat,
            timing=timing,
        )

    if not timing:
        # No config in the set produced a measured time -- the backend can't run at this
        # shape. Return None (not raise): infeasibility is expected, the caller skips it.
        tl_log(label, f"H={H} B={tune_batch}: no feasible config")
        return None

    # Every feasible config ranked by its measured tune-batch latency (ascending), so
    # callers can record the top-K rather than only the winner; ranked[0] is the winner.
    order = sorted(timing, key=lambda k: timing[k])
    ranked = [{"config": configs[i], "latency_ms": timing[i]} for i in order]
    best_cfg, best_ms = ranked[0]["config"], ranked[0]["latency_ms"]

    tl_log(
        label,
        f"H={H} B={tune_batch}: best {best_ms:.4f} ms "
        f"(rank #{order[0] + 1:>02}/{len(configs)}) [{_fmt_cfg(best_cfg)}]",
    )

    # Compile + re-validate + finalize the winner.
    kernel = factory.compile(**best_cfg)
    validate(kernel, best_cfg)
    return finalize(kernel, best_cfg), ranked
