#!/usr/bin/env python3
"""Sweep --joint_lattice_cap at a fixed total lattice-cell count and print the
cheapest cap within --threshold of the best throughput."""

import argparse
import math
from pathlib import Path

import torch
import yaml
from beartype import beartype
from beartype.typing import Callable, Tuple
from tilelang.profiler import do_bench

from caiman_asr_train.core.packing import packed_batch_offsets
from caiman_asr_train.rnnt import config
from caiman_asr_train.rnnt.loss import (
    IDENTITY_LOSS_MODIFIERS,
    FusedProjAndLoss,
    LossSpec,
    TransducerLossFn,
    build_loss_fn,
)
from caiman_asr_train.train_utils.core import autocast_if

SPM_SIZE = {"testing": 1023, "base": 8703, "large": 17407}


@beartype
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(SPM_SIZE))
    p.add_argument(
        "--config_dir", type=Path, default=Path("/workspace/training/configs")
    )
    p.add_argument("--cap_pow2", type=int, nargs="+", default=[17, 18, 19, 20, 21])
    p.add_argument("--utt_seconds", type=float, default=32.0)
    p.add_argument("--tokens_per_sec", type=float, default=3.0)
    p.add_argument("--total_lattice", type=int, default=2**24)
    p.add_argument("--threshold", type=float, default=0.97)
    p.add_argument("--transducer_backend", default="tilelang")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--n_repeat", type=int, default=3)
    return p.parse_args()


@beartype
def load_config(model: str, config_dir: Path) -> Tuple[dict, float]:
    path = config_dir / f"{model}-{SPM_SIZE[model]}sp.yaml"
    assert path.is_file(), f"no config at {path}; pass --config_dir"

    cfg = yaml.safe_load(path.read_text())
    rnnt_config = config.rnnt(cfg)
    feats = cfg["input_train"]["filterbank_features"]
    splice = cfg["input_train"]["frame_splicing"]
    frames_per_sec = (
        1.0
        / feats["window_stride"]
        / splice["frame_subsampling"]
        / rnnt_config["enc_stack_time_factor"]
    )

    return rnnt_config, frames_per_sec


@beartype
def build_loss(
    rnnt_config: dict, n_classes: int, backend: str, dtype: torch.dtype
) -> TransducerLossFn:
    spec = LossSpec(
        backend=backend,
        blank_idx=n_classes - 1,
        eos_idx=None,
        star_idx=None,
        packed_input=rnnt_config["joint_apex_transducer"] == "pack",
        joint_n_hid=rnnt_config["joint_n_hid"],
        device=torch.device("cuda"),
        dtype=dtype,
    )
    return build_loss_fn(spec)


@beartype
def make_cell(
    loss_fn: TransducerLossFn,
    proj: torch.nn.Linear,
    n_utts: int,
    t_len: int,
    u_len: int,
    n_splits: int,
    dtype: torch.dtype,
) -> Tuple[Callable[[], None], int]:
    n_hid, n_classes = proj.in_features, proj.out_features
    cells = n_utts * t_len * u_len
    a_buf = torch.relu(torch.randn(cells, n_hid, device="cuda", dtype=dtype))
    labels = torch.randint(
        0, n_classes - 1, (n_utts, u_len - 1), device="cuda", dtype=torch.int32
    )
    f_len = torch.full((n_utts,), t_len, device="cuda", dtype=torch.int32)
    y_len = torch.full((n_utts,), u_len - 1, device="cuda", dtype=torch.int32)
    boff = packed_batch_offsets(f_len, y_len + 1)
    fused = isinstance(loss_fn, FusedProjAndLoss)

    def run() -> None:
        proj.zero_grad(set_to_none=False)
        for _ in range(n_splits):
            a = a_buf.detach().requires_grad_(True)
            with autocast_if(dtype is not torch.float32):
                if fused:
                    loss = loss_fn(
                        a,
                        proj.weight.t(),
                        proj.bias,
                        labels,
                        f_len,
                        y_len,
                        boff,
                        cells,
                        IDENTITY_LOSS_MODIFIERS,
                    ).mean()
                else:
                    loss = loss_fn(
                        proj(a),
                        f_len,
                        labels,
                        y_len,
                        batch_offset=boff,
                        max_f_len=t_len,
                        loss_mods=IDENTITY_LOSS_MODIFIERS,
                    )
            (loss / n_splits).backward()
            # The fused backward parks a cells-by-vocab tensor on the autograd
            # ctx, so the graph must die before the next forward.
            del loss, a

    return run, cells


@beartype
def select(throughput: dict[int, float], threshold: float) -> int:
    best = max(throughput, key=throughput.__getitem__)
    return min(
        c
        for c in throughput
        if c <= best and throughput[c] >= threshold * throughput[best]
    )


@beartype
def main() -> None:
    args = parse_args()
    torch.manual_seed(1)
    dtype = torch.float32 if args.no_amp else torch.float16

    n_classes = SPM_SIZE[args.model] + 1
    rnnt_config, frames_per_sec = load_config(args.model, args.config_dir)
    n_hid = rnnt_config["joint_n_hid"]
    loss_fn = build_loss(rnnt_config, n_classes, args.transducer_backend, dtype)
    proj = torch.nn.Linear(n_hid, n_classes).cuda()

    t_len = max(1, math.ceil(args.utt_seconds * frames_per_sec))
    u_len = max(2, round(args.utt_seconds * args.tokens_per_sec) + 1)
    print(
        f"{args.model}: V={n_classes} joint_n_hid={n_hid} "
        f"T={t_len} U={u_len} cells/utt={t_len * u_len}",
        flush=True,
    )

    throughput: dict[int, float] = {}
    for pow2 in sorted(args.cap_pow2):
        n_utts = max(1, 2**pow2 // (t_len * u_len))
        n_splits = max(1, round(args.total_lattice / (n_utts * t_len * u_len)))
        try:
            run, cells = make_cell(loss_fn, proj, n_utts, t_len, u_len, n_splits, dtype)
            torch.cuda.reset_peak_memory_stats()
            ms = do_bench(
                run, _n_warmup=1, _n_repeat=args.n_repeat, return_mode="median"
            )
            mem = torch.cuda.max_memory_allocated() / 2**30
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  cap 2^{pow2}  OOM", flush=True)
            break
        throughput[pow2] = cells * n_splits / (ms / 1e3)
        print(
            f"  cap 2^{pow2}  utts/split={n_utts:<5} splits={n_splits:<5} "
            f"{throughput[pow2] / 1e6:7.2f} Mcells/s  {mem:5.2f} GiB",
            flush=True,
        )
        del run
        torch.cuda.empty_cache()

    assert throughput, "every cap OOMed; lower --utt_seconds or --total_lattice"
    pick = select(throughput, args.threshold)
    print(f"\njoint_lattice_cap = {2**pick}  (2^{pick}, {args.threshold:.0%} rule)")


if __name__ == "__main__":
    main()
