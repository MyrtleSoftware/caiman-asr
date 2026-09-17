"""
Sweep the current GPU and write the two data files this package uses: the per-config
TileLang ranking the tree model trains on (``data/full_ranked_<gpu>.jsonl.gz``) and the
CustomLSTM baseline the plots overlay (``data/custom_lstm_<gpu>.jsonl.gz``). Run once per
GPU to add it to the training set.

``--custom-only`` regenerates just the baseline; ``--skip-custom`` writes only the TileLang
sweep. Run inside the GPU training container (needs CUDA + the tilelang stack)::

    docker run --rm --gpus all -e PYTHONPATH=/workspace/training \\
        myrtle/rnnt python -m caiman_asr_train.tilelstm.tree_model.core.gen_data

Before a full multi-H sweep read the README's "Running a healthy full sweep
(operations)" section: raise ``vm.max_map_count``, keep
``CAIMAN_LSTM_PAR_COMPILE_WORKERS`` low (4-8), and on a multi-GPU host split by
``--tune-batches`` (not ``--backends``) with a distinct ``--tag`` per agent --
train.py globs + merges the per-agent files.
"""

import argparse
import gzip
import json
import math
import os
import re
from collections import deque
from dataclasses import asdict

import torch
from beartype import beartype

from caiman_asr_train.autotune.bench import lean_bench, n_repeat_for
from caiman_asr_train.autotune.dtypes import tl_dtype
from caiman_asr_train.tilelstm.autotune.kernel import Kernel
from caiman_asr_train.tilelstm.autotune.select import _SPLITK_MAX_BATCH
from caiman_asr_train.tilelstm.autotune.shapes import PROFILE_SEQ, tune_batch_for
from caiman_asr_train.tilelstm.autotune.tuners import tune
from caiman_asr_train.tilelstm.tree_model import BACKENDS, DATA_DIR
from caiman_asr_train.tilelstm.tree_model.core.specs import host_spec

# Regime cut-offs so each backend is benched only where select.py picks it: split-K below
# _SPLITK_MAX_BATCH, persistent at/above the per-H coop residency wall (coop self-skips past
# its wall, where tune returns None). Err low on the floor so the crossover is covered.
PERSISTENT_MIN_B = {1024: 256, 1536: 128}


@beartype
def _log_refine_order(values: list[int]) -> list[int]:
    """
    Reorder a sorted list of B values coarse-to-fine by log-space bisection: the two
    endpoints first, then the value nearest each interval's geometric midpoint, recursing
    on the halves, so every prefix is ~log-uniform and an interrupted sweep still covers
    [min, max].
    """
    items = sorted(set(values))
    n = len(items)
    if n <= 2:
        return items
    logs = [math.log(v) for v in items]
    used = {0, n - 1}
    order = [items[0], items[n - 1]]
    q = deque([(0, n - 1)])
    while q:
        lo, hi = q.popleft()
        if hi - lo <= 1:
            continue
        target = (logs[lo] + logs[hi]) / 2.0  # geometric midpoint of the interval
        cand = min(
            (i for i in range(lo + 1, hi) if i not in used),
            key=lambda i: abs(logs[i] - target),
            default=None,
        )
        if cand is None:
            continue
        used.add(cand)
        order.append(items[cand])
        q.append((lo, cand))
        q.append((cand, hi))
    return order


@beartype
def progressive_buckets(b_max: int = 1536) -> list[int]:
    """
    The distinct ``tune_batch_for`` buckets in [1, b_max], ordered coarse-to-fine by
    log-space bisection (``_log_refine_order``) so the sweep refines progressively and stays
    interruptible with log-uniform coverage.
    """
    return _log_refine_order(tune_buckets(list(range(1, b_max + 1))))


# Stable filename ids for the committed sweeps, so re-sweeping a known host overwrites its
# file instead of dropping a second slug-named sweep of the same GPU beside it (train would
# then double-count its shapes).
_FILE_IDS = (("4090", "4090"), ("A100", "a100"), ("RTX PRO 6000", "sm120"))


@beartype
def device_id(name: str) -> str:
    """
    Short, stable filename id for a device name: the committed ids where known, else a
    filesystem-safe slug of the raw name.
    """
    for sub, gid in _FILE_IDS:
        if sub in name:
            return gid
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


@beartype
def tune_buckets(ladder: list[int]) -> list[int]:
    """
    Distinct, sorted ``tune_batch`` buckets the ``ladder`` of production batches maps
    to. The sweep is keyed by bucket (the tuned binary is batch-invariant within a
    bucket), so we only tune each bucket once.
    """
    return sorted({tune_batch_for(b) for b in ladder})


@beartype
def sweep(
    backends: list[Kernel],
    hiddens: list[int],
    buckets: list[int],
    device: torch.device,
    lo: torch.dtype,
    hi: torch.dtype,
    out_path: str,
    n_warmup: int,
    n_repeat: int | None,
) -> tuple[int, int]:
    """
    Tune every ``(backend, H, tune_batch)`` shape and stream the rankings to
    ``out_path`` as gzipped JSONL. Returns ``(n_shapes, n_configs)`` actually written.

    The gzip stream is flushed after the header and after every shape, so an interrupted
    run keeps the shapes it already wrote (read it with ``zcat`` / the loader's gzip
    opener). A shape that is infeasible for a backend is skipped (one-line note, no
    record); any other error on a shape is logged and skipped too, so one bad shape never
    stalls the whole sweep.
    """
    header = {
        "record": "header",
        "device": torch.cuda.get_device_name(device),
        # The sweep host's measured spec, so the file is self-describing (train featurizes
        # on it, the bundle carries it forward).
        "spec": asdict(host_spec(device.index if device.index is not None else 0)),
        "kind": "full_ranked",
        "lo_dtype": tl_dtype(lo),
        "hi_dtype": tl_dtype(hi),
        "profile_seq": PROFILE_SEQ,
        "backends": list(backends),
        "source": "gen_data.py tune() full ranking",
    }
    n_shapes = n_configs = 0
    with gzip.open(out_path, "wt") as fh:
        fh.write(json.dumps(header, default=str) + "\n")
        fh.flush()
        # B is the outer loop in coarse-to-fine order, so an interrupted run still has
        # log-uniform B coverage of every backend.
        for tb in buckets:
            for H in hiddens:
                for backend in backends:
                    # Regime filter: bench each backend only where select.py would pick it.
                    if backend.is_splitk() and tb >= _SPLITK_MAX_BATCH:
                        continue
                    if backend.is_persistent() and tb < PERSISTENT_MIN_B.get(
                        H, max(PERSISTENT_MIN_B.values())
                    ):
                        continue
                    try:
                        tuned = tune(
                            backend,
                            H,
                            device,
                            tb,
                            lo,
                            hi,
                            n_warmup=n_warmup,
                            n_repeat=n_repeat,
                        )
                    except Exception as err:  # noqa: BLE001
                        print(f"  ERROR: {backend} H={H} B={tb}: {err!r}")
                        continue
                    if tuned is None:  # no feasible config at this shape
                        print(f"  skip (infeasible): {backend} H={H} B={tb}")
                        continue
                    _, ranked = tuned
                    rec = {
                        "record": "shape",
                        "backend": backend,
                        "H": H,
                        "tune_batch": tb,
                        "ranked": ranked,
                    }
                    fh.write(json.dumps(rec, default=str) + "\n")
                    fh.flush()
                    n_shapes += 1
                    n_configs += len(ranked)
                    print(
                        f"  {backend} H={H} B={tb}: {len(ranked)} configs "
                        f"(best {ranked[0]['latency_ms']:.4f} ms)"
                    )
    return n_shapes, n_configs


@beartype
def bench_custom(
    H: int, B: int, device: torch.device, n_warmup: int, n_repeat: int | None
) -> dict[str, float]:
    """
    Median ms of the CustomLSTM fused forward and backward recurrent kernels at (H, B) --
    the recurrent-scan equivalents of the TileLang ``lstm_fwd``/``lstm_bwd`` kernels,
    benched as the dashed baseline the throughput plots overlay.

    Buffers mirror ``rnnt_ext.custom_lstm.lstm.Function`` (fp16, the model's autocast
    operand dtype): recurrent weight ``Rp`` is ``[4H, H]`` (weight_hh), the input-projection
    gates are ``[S, B, 4H]``, and the hidden/cell buffers ``[S+1, B, H]`` carry the seed in
    slot 0. The forward overwrites ``gates`` in place with the post-activation gates the
    backward reads (the aliasing the TileLang scan also uses), so we run it once before
    timing backward. Returns ``{"cust_fwd": ms, "cust_bwd": ms}``.
    """
    import rnnt_ext.cuda.lstm as lstm_cu  # lazy: keep the module importable without the ext

    S = PROFILE_SEQ
    dt = torch.float16
    rep = n_repeat if n_repeat is not None else n_repeat_for(B)
    rsh = 1.0 / (H**0.5)  # small magnitudes so the 64-step cell scan stays finite

    def rand(*shape):
        return (torch.rand(*shape, device=device, dtype=dt) * 2.0 - 1.0) * rsh

    Rp = rand(4 * H, H)
    gates = rand(S, B, 4 * H)
    c = torch.zeros(S + 1, B, H, device=device, dtype=dt)
    y = torch.zeros(S + 1, B, H, device=device, dtype=dt)

    fwd_ms = lean_bench(
        lambda: lstm_cu.lstm_fused_fwd_soft(Rp, gates, c, y), n_warmup, rep
    )

    # Populate the post-activation gates + cell states the backward kernel consumes.
    lstm_cu.lstm_fused_fwd_soft(Rp, gates, c, y)
    delta = rand(S, B, H)
    dG = torch.empty(S, B, 4 * H, device=device, dtype=dt)
    bwd_ms = lean_bench(
        lambda: lstm_cu.lstm_fused_bwd_soft(Rp, gates, c, delta, dG), n_warmup, rep
    )

    return {"cust_fwd": fwd_ms, "cust_bwd": bwd_ms}


@beartype
def sweep_custom(
    hiddens: list[int],
    buckets: list[int],
    device: torch.device,
    out_path: str,
    n_warmup: int,
    n_repeat: int | None,
) -> int:
    """
    Bench the CustomLSTM fused kernels at every ``(H, B)`` and stream the latencies to
    ``out_path`` as gzipped JSONL in the same shape-record schema as ``sweep`` (synthetic
    backends ``cust_fwd`` / ``cust_bwd``), so ``aux.plot`` reads it unchanged while the tree
    glob (``full_ranked_*``) ignores it. Returns the number of shape records written.
    """
    header = {
        "record": "header",
        "device": torch.cuda.get_device_name(device),
        "kind": "custom_lstm",
        "lo_dtype": "float16",
        "hi_dtype": "float32",
        "profile_seq": PROFILE_SEQ,
        "backends": ["cust_fwd", "cust_bwd"],
        "source": "gen_data.py lstm_cu.lstm_fused_*_soft",
    }
    n = 0
    with gzip.open(out_path, "wt") as fh:
        fh.write(json.dumps(header) + "\n")
        fh.flush()
        for B in buckets:
            for H in hiddens:
                try:
                    res = bench_custom(H, B, device, n_warmup, n_repeat)
                except Exception as err:  # noqa: BLE001
                    print(f"  ERROR cust H={H} B={B}: {err!r}")
                    continue
                for backend, ms in res.items():
                    rec = {
                        "record": "shape",
                        "backend": backend,
                        "H": H,
                        "tune_batch": B,
                        "ranked": [{"config": {"H": H, "B": B}, "latency_ms": ms}],
                    }
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    n += 1
                    print(f"  {backend} H={H} B={B}: {ms:.4f} ms")
    return n


@beartype
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[1024, 1536],
        help="hidden sizes to sweep (default: 1024 1536)",
    )
    parser.add_argument(
        "--tune-batches",
        type=int,
        nargs="+",
        default=None,
        help="explicit B values to sweep, used verbatim (overrides the default progressive "
        "tune_batch_for buckets); still walked in log-refine order so a finer manual grid "
        "stays interruptible",
    )
    parser.add_argument(
        "--b-max",
        type=int,
        default=1536,
        help="upper batch bound for the default progressive bucket sweep (default: 1536); "
        "ignored when --tune-batches is given. Raising it (e.g. 2048) extends coverage to "
        "larger batches; a re-run replays the cached lower buckets and only benches the "
        "new high-B buckets",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index to run on (default: 0). On a multi-GPU host, launch one "
        "agent per GPU with disjoint --backends, each with its own --device i",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="output-filename suffix so parallel agents (disjoint --backends / different "
        "GPUs on one host) don't clobber; defaults to the sorted backend subset when a "
        "strict subset is given (no suffix for the full set)",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=3,
        help="lean-bench warm-up launches (default: 3)",
    )
    parser.add_argument(
        "--n-repeat",
        type=int,
        default=None,
        help="lean-bench timed launches per config; median is taken. Default: auto "
        "per-batch (engine.n_repeat_for -- ~768*10/tune_batch, clamped to [10, 256]). Pass "
        "an int to force a constant across all batches.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        type=Kernel,
        choices=BACKENDS,
        default=list(BACKENDS),
        help="backends to tune (default: all 7)",
    )
    parser.add_argument(
        "--out-dir",
        default=DATA_DIR,
        help="output directory (default: this package's data/ dir)",
    )
    custom = parser.add_mutually_exclusive_group()
    custom.add_argument(
        "--skip-custom",
        action="store_true",
        help="do not bench the CustomLSTM baseline (write only the TileLang full_ranked "
        "sweep). By default gen_data also writes custom_lstm_<gpu>.jsonl.gz for the plots, "
        "except when a strict --backends subset is selected (parallel-agent mode), where "
        "the baseline is auto-skipped so agents don't each re-bench it.",
    )
    custom.add_argument(
        "--custom-only",
        action="store_true",
        help="bench only the CustomLSTM baseline (skip the TileLang sweep). Use once to "
        "(re)generate custom_lstm_<gpu>.jsonl.gz -- e.g. after a multi-GPU disjoint "
        "--backends run, where the baseline was auto-skipped.",
    )
    return parser.parse_args(argv)


@beartype
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("gen_data.py requires CUDA: no GPU is available")

    torch.set_grad_enabled(False)
    device = torch.device(f"cuda:{args.device}")
    lo, hi = torch.float16, torch.float32
    name = torch.cuda.get_device_name(device)

    if args.tune_batches is not None:
        buckets = _log_refine_order(sorted(set(args.tune_batches)))
    else:
        buckets = progressive_buckets(b_max=args.b_max)

    # A strict --backends subset is parallel-agent mode. The CustomLSTM baseline has no
    # backend dimension, so it's benched once (full run or --custom-only), not per agent.
    subset = set(args.backends) != set(BACKENDS)
    run_tilelang = not args.custom_only
    run_custom = args.custom_only or (not args.skip_custom and not subset)

    os.makedirs(args.out_dir, exist_ok=True)
    n_repeat_desc = (
        "auto (~768*10/B, clamped [10, 256])"
        if args.n_repeat is None
        else str(args.n_repeat)
    )
    print(f"device   : cuda:{args.device} {name}")
    print(f"hidden   : {args.hidden}")
    print(f"buckets  : {len(buckets)} B (progressive log-refine order)")
    print(f"bench    : lean (n_warmup={args.n_warmup}, n_repeat={n_repeat_desc})")

    if run_tilelang:
        # Per-agent filename suffix so parallel agents write distinct files; train globs
        # full_ranked_* and merges.
        tag = args.tag or ("-".join(sorted(args.backends)) if subset else None)
        suffix = f"_{tag}" if tag else ""
        tl_path = os.path.join(
            args.out_dir, f"full_ranked_{device_id(name)}{suffix}.jsonl.gz"
        )
        print("\n== TileLang autotuner sweep ==")
        print(f"backends : {args.backends}")
        print(f"out      : {tl_path}\n")
        n_shapes, n_configs = sweep(
            args.backends,
            args.hidden,
            buckets,
            device,
            lo,
            hi,
            tl_path,
            args.n_warmup,
            args.n_repeat,
        )
        print(f"\nwrote {tl_path}")
        print(f"  {n_shapes} shape records, {n_configs} configs total")

    if run_custom:
        # The baseline has no backend split, so it takes only an explicit --tag (a
        # --custom-only rerun then lands on the shared file).
        csuffix = f"_{args.tag}" if args.tag else ""
        cust_path = os.path.join(
            args.out_dir, f"custom_lstm_{device_id(name)}{csuffix}.jsonl.gz"
        )
        print("\n== CustomLSTM baseline sweep ==")
        print(f"out      : {cust_path}\n")
        n_cust = sweep_custom(
            args.hidden, buckets, device, cust_path, args.n_warmup, args.n_repeat
        )
        print(f"\nwrote {cust_path}")
        print(f"  {n_cust} shape records")
    elif subset and not args.skip_custom:
        print(
            "\n(CustomLSTM baseline auto-skipped: a --backends subset was selected -- run "
            "`--custom-only` once to (re)generate custom_lstm_<gpu>.jsonl.gz)"
        )


if __name__ == "__main__":
    main()
