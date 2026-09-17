"""
Plot achieved compute throughput (TFLOP/s) vs batch from the autotune sweep data.

Reads the per-GPU ``data/full_ranked_<gpu>[_<tag>].jsonl.gz`` files from ``gen_data`` (a
GPU's data may span several files, all merged) and plots each forward/backward kernel's
best-tuned config at every ``tune_batch`` bucket -- one subplot per hidden size, log-2 batch
axis, with the CustomLSTM baseline (``custom_lstm_<gpu>.jsonl.gz``) overlaid dashed. One
figure per GPU per direction. Throughput uses the recurrent-gemm FLOP count 8*B*H**2*S.

Run in the container (CPU-only)::

    python -m caiman_asr_train.tilelstm.tree_model.aux.plot --out-dir DIR
    python -m caiman_asr_train.tilelstm.tree_model.aux.plot --data-dir DATA --out-dir DIR

Narrow with ``--gpu 4090`` / ``--direction fwd``; ``--best-only`` collapses each panel to
the fastest backend at each B and writes ``*_best.png``.
"""

import argparse
import glob
import gzip
import json
import os
from collections import defaultdict

import matplotlib
import numpy as np
from beartype import beartype

# Only for the --data-dir default: the datasets-drive location when populated, else the
# in-tree data/ beside the tree model.
from caiman_asr_train.tilelstm.tree_model import read_data_dir

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

# Kernel -> (legend label, matplotlib color).
FWD = [
    ("fwd_c", "coop", "C0"),
    ("fwd_p", "persistent", "C1"),
    ("fwd_ct", "coop_train", "C2"),
    ("fwd_pt", "persistent_train", "C3"),
    ("cust_fwd", "CustomLSTM", "k"),  # dashed baseline overlay
]
BWD = [
    ("bwd_c", "coop", "C0"),
    ("bwd_p", "persistent", "C1"),
    ("bwd_s", "splitk", "C2"),
    ("cust_bwd", "CustomLSTM", "k"),  # dashed baseline overlay
]
DIRS = {
    "fwd": ("FORWARD", 8, "8", FWD),
    "bwd": ("BACKWARD", 8, "8", BWD),
}
# --best-only: collapse each panel to the per-B upper envelope (fastest backend at each B).
# Forward keeps inference and training as separate envelopes; CustomLSTM stays a dashed
# full-B reference, not folded in.
BEST = {
    "fwd": [
        ("best (inference)", ("fwd_c", "fwd_p"), "C0"),
        ("best (train)", ("fwd_ct", "fwd_pt"), "C3"),
    ],
    "bwd": [
        ("best", ("bwd_c", "bwd_p", "bwd_s"), "C0"),
    ],
}
# Fallback only -- the H panels are discovered from each GPU's data (see plot_direction).
HIDDENS = [512, 768, 1024, 1536]

# The merged sweep series: (backend, H) -> {tune_batch: best latency ms}.
Best = dict[tuple[str, int], dict[int, float]]


@beartype
def _split_scale(ax: Axes, bmax: int, split: int) -> None:
    """
    Split x-scale: log2 for B < ``split``, linear for B >= ``split``.

    A matplotlib ``function`` scale whose piecewise forward/inverse is C1-continuous at
    the join -- the linear slope equals d/dB[log2(B)] at ``split`` -- so the two regions
    meet with no kink. Ticks are powers of two in the log region and ``2*split`` multiples
    in the linear region; a faint dotted line marks the join.
    """
    split = float(split)
    log_s = float(np.log2(split))
    slope = 1.0 / (split * float(np.log(2.0)))  # d/dB log2(B) at B = split

    def fwd(b):
        b = np.asarray(b, dtype=float)
        return np.where(
            b <= split, np.log2(np.clip(b, 1e-12, None)), log_s + (b - split) * slope
        )

    def inv(x):
        x = np.asarray(x, dtype=float)
        return np.where(x <= log_s, np.power(2.0, x), split + (x - log_s) / slope)

    ax.set_xscale("function", functions=(fwd, inv))
    log_ticks = [t for t in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024) if t <= split]
    step = int(split) * 2
    ticks = log_ticks + list(range(step, int(bmax) + 1, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(1, bmax)
    ax.axvline(split, color="0.7", linestyle=":", linewidth=0.8, zorder=0)


@beartype
def load(path: str) -> tuple[str, int, Best]:
    """
    Return ``(device, profile_seq, {(backend, H): {B: best_latency_ms}})``.

    Tolerant of a still-being-written gzip tail (stops at the last complete line).
    """
    device, seq = os.path.basename(path), 64
    best = {}
    try:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    break  # truncated final line of a live file
                if r.get("record") == "header":
                    # ``or`` so a present-but-null field falls back too.
                    device = r.get("device") or device
                    seq = int(r.get("profile_seq") or seq)
                elif r.get("record") == "shape":
                    lat = min(e["latency_ms"] for e in r["ranked"])
                    best[(r["backend"], r["H"])] = best.get((r["backend"], r["H"]), {})
                    best[(r["backend"], r["H"])][r["tune_batch"]] = lat
    except (OSError, EOFError) as err:
        print(f"  warning: {path}: {err}")
    return device, seq, best


@beartype
def _header_device(path: str) -> str | None:
    """
    The sweep file's header ``device`` string, or ``None`` if it has no header / can't be
    read. Grouping by the header device (not a filename token) avoids conflating GPUs whose
    ``device_id`` slug contains underscores with the optional ``_<tag>`` suffix.
    """
    try:
        with gzip.open(path, "rt") as fh:
            first = fh.readline()  # the header is always the first record
    except (OSError, EOFError):
        return None
    try:
        r = json.loads(first)
    except json.JSONDecodeError:
        return None
    return r.get("device") if r.get("record") == "header" else None


@beartype
def merge_files(paths: list[str]) -> tuple[str | None, int, Best]:
    """
    Merge one GPU's data spread across files into one ``(device, seq, best)``: every
    ``full_ranked_*`` file for the GPU plus an optional ``custom_lstm_*`` baseline.
    Device/seq come from the TileLang files (the CustomLSTM file shares GPU + seq).
    """
    device, seq, best = None, 64, {}
    for p in sorted(paths):
        d, s, b = load(p)
        if "full_ranked" in os.path.basename(p):
            device, seq = d, s
        for k, series in b.items():
            best.setdefault(k, {}).update(series)
    return device, seq, best


@beartype
def _envelope(best: Best, members: tuple[str, ...], H: int) -> dict[int, float]:
    """
    Per-B fastest latency across ``members`` at hidden size ``H``: ``{B: min latency}`` over
    the members that have data at that B. Plotted, this is the upper throughput envelope --
    whichever backend is fastest at each batch.
    """
    merged = {}
    for bk in members:
        for B, lat in best.get((bk, H), {}).items():
            if B not in merged or lat < merged[B]:
                merged[B] = lat
    return merged


@beartype
def _plot_series(
    ax: Axes,
    series: dict[int, float],
    H: int,
    factor: int,
    seq: int,
    *,
    color: str,
    label: str,
    dashed: bool,
) -> None:
    """
    Plot one throughput-vs-B line (TFLOP/s) for ``series`` (``{B: latency_ms}``). Dashed
    marks the CustomLSTM baseline; solid ``.-`` a benched TileLang line/best-only envelope.
    """
    bs = sorted(series)
    tput = [factor * B * H * H * seq / (series[B] * 1e-3) / 1e12 for B in bs]
    if dashed:
        ax.plot(
            bs, tput, color=color, label=label, linestyle="--", linewidth=1.6, zorder=3
        )
    else:
        ax.plot(bs, tput, ".-", color=color, label=label, markersize=4, linewidth=1.2)


@beartype
def plot_direction(
    device: str,
    seq: int,
    best: Best,
    direction: str,
    out_dir: str,
    split: int,
    best_only: bool = False,
) -> str | None:
    title_word, factor, flop_str, backends = DIRS[direction]
    # One panel per H present in this GPU's data (sorted); the CustomLSTM line is overlaid
    # dashed/black. sharey lets a kernel be read straight across panels.
    hiddens = sorted({H for (_bk, H) in best}) or HIDDENS
    fig, axes = plt.subplots(
        1, len(hiddens), figsize=(7.5 * len(hiddens), 5.5), sharey=True, squeeze=False
    )
    axes = axes[0]
    bmax = max((B for s in best.values() for B in s), default=1024)

    any_data = False
    for ax, H in zip(axes, hiddens):
        if best_only:
            # One solid envelope per group (fastest backend at each B); forward keeps
            # the train and inference paths as two separate lines.
            for label, members, color in BEST[direction]:
                series = _envelope(best, members, H)
                if series:
                    _plot_series(
                        ax,
                        series,
                        H,
                        factor,
                        seq,
                        color=color,
                        label=label,
                        dashed=False,
                    )
                    any_data = True
            # plus the CustomLSTM dashed reference (full-B).
            for backend, label, color in backends:
                series = best.get((backend, H))
                if backend.startswith("cust") and series:
                    _plot_series(
                        ax,
                        series,
                        H,
                        factor,
                        seq,
                        color=color,
                        label=label,
                        dashed=True,
                    )
                    any_data = True
        else:
            for backend, label, color in backends:
                series = best.get((backend, H))
                if not series:
                    continue
                _plot_series(
                    ax,
                    series,
                    H,
                    factor,
                    seq,
                    color=color,
                    label=label,
                    dashed=backend.startswith("cust"),
                )
                any_data = True
        _split_scale(ax, bmax, split)
        ax.set_xlabel("batch size (B)")
        ax.set_title(f"H = {H}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper left")
    axes[0].set_ylabel(
        f"achieved tensor throughput (TFLOP/s, recurrent gemm = {flop_str}·B·H²·S)"
    )
    note = "  ·  fastest backend per B" if best_only else ""
    fig.suptitle(
        f"LSTM fused-scan {title_word} compute throughput vs batch  ({device}){note}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if not any_data:
        print(f"  no {direction} data for {device}; skipping")
        plt.close(fig)
        return None
    safe = "".join(c if c.isalnum() else "_" for c in device)
    suffix = "_best" if best_only else ""
    out = os.path.join(out_dir, f"lstm_kernel_{direction}_flops_{safe}{suffix}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


@beartype
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--data-dir",
        default=read_data_dir(),
        help="dir of full_ranked_*.jsonl.gz (default: /datasets/tile_tree_data when it "
        "holds sweeps, else the in-tree data/)",
    )
    p.add_argument(
        "--gpu",
        nargs="+",
        default=None,
        help="GPU tags to plot (e.g. 4090 a100); default: every full_ranked_* file found",
    )
    p.add_argument(
        "--direction",
        choices=["fwd", "bwd", "both"],
        default="both",
        help="forward, backward, or both (default: both)",
    )
    p.add_argument("--out-dir", default=".", help="where to write PNGs (default: cwd)")
    p.add_argument(
        "--split-at",
        type=int,
        default=256,
        help="batch where the x-axis switches from log (below) to linear (above) "
        "(default: 256)",
    )
    p.add_argument(
        "--best-only",
        action="store_true",
        help="plot only the fastest backend at each B (a per-B upper envelope) instead of "
        "every backend; forward keeps the inference and training paths as separate lines, "
        "with the CustomLSTM dashed reference kept across all B. Writes *_best.png.",
    )
    args = p.parse_args(argv)

    tl_files = sorted(glob.glob(os.path.join(args.data_dir, "full_ranked_*.jsonl.gz")))
    cust_files = sorted(
        glob.glob(os.path.join(args.data_dir, "custom_lstm_*.jsonl.gz"))
    )
    if not tl_files:
        raise SystemExit(f"no full_ranked_*.jsonl.gz in {args.data_dir}")

    # Group every sweep file by its header device, so a GPU spanning several tagged files is
    # merged and two GPUs are never conflated by an ambiguous filename token. CustomLSTM
    # files overlay only a device that has TileLang data.
    groups = defaultdict(list)
    for f in tl_files:
        groups[_header_device(f) or os.path.basename(f)].append(f)
    for f in cust_files:
        dev = _header_device(f) or os.path.basename(f)
        if dev in groups:
            groups[dev].append(f)

    devices = sorted(groups)
    if args.gpu:
        # Match each --gpu token as a case-insensitive substring of the device name or any
        # filename, so 'sm120' (absent from the marketing name) resolves via its filename.
        want = [g.lower() for g in args.gpu]
        devices = [
            d
            for d in devices
            if any(w in d.lower() for w in want)
            or any(w in os.path.basename(f).lower() for f in groups[d] for w in want)
        ]
    if not devices:
        raise SystemExit(
            f"no full_ranked_*.jsonl.gz in {args.data_dir} matching {args.gpu}"
        )
    os.makedirs(args.out_dir, exist_ok=True)
    directions = ["fwd", "bwd"] if args.direction == "both" else [args.direction]

    for dev in devices:
        device, seq, best = merge_files(groups[dev])
        print(f"{device}")
        for d in directions:
            try:
                plot_direction(
                    device, seq, best, d, args.out_dir, args.split_at, args.best_only
                )
            except Exception as err:  # noqa: BLE001
                print(f"  ERROR: {d} {device}: {err!r}")


if __name__ == "__main__":
    main()
