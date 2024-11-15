#!/usr/bin/env python3
import csv
import gc
import json
from argparse import ArgumentParser, Namespace
from copy import copy
from pathlib import Path

import torch

from caiman_asr_train.args.val import check_val_arguments
from caiman_asr_train.evaluate.error_rates import ErrorRate, error_rate_abbrev
from caiman_asr_train.setup.val import ValCPUSetup, ValSetup
from caiman_asr_train.train_utils.torchrun import maybe_restart_with_torchrun
from caiman_asr_train.val import check_shared_args, val_arg_parser, validate


def add_val_multiple_args_in_place(parser: ArgumentParser) -> None:
    val_multiple = parser.add_argument_group("Validate multiple args")
    val_multiple.add_argument(
        "--all_dataset_dirs",
        "--all_data_dirs",
        help=(
            "A list of the data directories of the validation datasets. Must be "
            "same length as '--all_val_manifests'. NOTE that the --dataset_dir arg "
            "will have no effect in this script"
        ),
        nargs="+",
        required=True,
    )
    val_multiple.add_argument(
        "--all_val_manifests",
        help=(
            "A list of the validation dataset manifests. Must be the same length "
            "as '--all_dataset_dirs'. NOTE that the --val_manifests arg will have no "
            "effect in this script"
        ),
        nargs="+",
        required=True,
    )
    val_multiple.add_argument(
        "--custom_batch_sizes",
        help=(
            "A list of the batch sizes for each dataset, overriding '--val_batch_size'. "
            "Must be the same length as '--all_dataset_dirs'. "
            "If unset, all datasets will default to '--val_batch_size'."
        ),
        nargs="+",
        type=int,
        default=None,
    )
    val_multiple.add_argument(
        "--overwrite_ok",
        action="store_true",
        default=False,
        help=(
            "If passed, allow overwriting of results saved to "
            "<--output_dir>/validate_multiple.json"
        ),
    )


def check_val_multiple_args(args: Namespace):
    if args.num_gpus > 1:
        assert not args.cpu, "Cannot use --cpu with multiple processes"

    if len(args.all_val_manifests) != len(args.all_dataset_dirs):
        raise ValueError(
            f"'--all_val_manifests'={args.all_val_manifests} and "
            f"'--all_dataset_dirs'={args.all_dataset_dirs} must be the same length"
        )
    if args.custom_batch_sizes is not None and len(args.custom_batch_sizes) != len(
        args.all_dataset_dirs
    ):
        raise ValueError(
            f"'--custom_batch_sizes'={args.custom_batch_sizes} and "
            f"'--all_dataset_dirs'={args.all_dataset_dirs} must be the same length"
        )


def validate_multiple(args: Namespace):
    """
    Validate args.checkpoint over multiple datasets.

    The results will be saved to "{args.output_dir}/validate_multiple.json"
    """
    results_fp = Path(f"{args.output_dir}/validate_multiple.json")
    results_csv_fp = Path(f"{args.output_dir}/validate_multiple.csv")
    if results_fp.exists() and not args.overwrite_ok:
        raise ValueError(
            f"Attempting to overwrite {results_fp}. Either pass a new --output_dir "
            "or, if overwriting is the intended behaviour, pass --overwrite_ok"
        )
    results_fp.parent.mkdir(exist_ok=True)

    setup_class = ValCPUSetup if args.cpu else ValSetup
    all_results = {}
    skip_init = False
    all_batch_sizes = (
        [args.val_batch_size] * len(args.all_dataset_dirs)
        if args.custom_batch_sizes is None
        else args.custom_batch_sizes
    )

    def _inner(manifest, dataset_dir, batch_size, skip_init) -> ErrorRate:
        """
        Pythons tightest scope is a function
        """

        val_name = Path(manifest).with_suffix("").name
        output_dir = Path(args.output_dir) / val_name
        output_dir.mkdir(exist_ok=True)

        val_args = copy(args)
        val_args.val_manifests = [manifest]
        val_args.dataset_dir = dataset_dir
        val_args.val_batch_size = batch_size
        val_args.output_dir = str(output_dir)
        val_args.skip_init = skip_init

        val_objects = setup_class().run(val_args)
        model_results = validate(val_args, val_objects=val_objects)

        manifest_path = Path(val_args.dataset_dir) / manifest

        all_results[str(manifest_path)] = model_results

        return val_objects.error_rate

    for manifest, dataset_dir, batch_size in zip(
        args.all_val_manifests, args.all_dataset_dirs, all_batch_sizes, strict=True
    ):
        error_rate = _inner(manifest, dataset_dir, batch_size, skip_init)

        # Only initialise logging once
        skip_init = True

        # Force cleanup to reduce fragmentation between evaluations.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    abbrev = error_rate_abbrev(error_rate)

    out_keys = [
        abbrev,
        "eos_frac",
        "sil_frac",
        "rem_frac",
        "latency_metrics",
    ]

    out_json = {
        val_dataset: {
            metric: val_results[metric] for metric in out_keys if metric in val_results
        }
        for val_dataset, val_results in all_results.items()
    }

    out_json["args"] = args.__dict__

    with results_fp.open("w") as f:
        json.dump(out_json, f, indent=2)

    write_csv(abbrev, results_csv_fp, all_results)

    return all_results


def write_csv(abbrev, results_csv_fp, all_results):
    #
    def nice(x):
        if isinstance(x, float):
            return f"{x:.4f}"

        return x

    def to_row(metric: str, proj):
        """`metric` is a string like "WER".
        proj is a function that takes all the results for
        one dataset and selects the `metric` result"""
        return {
            "Metric": metric,
            **{
                val_dataset: nice(proj(val_results))
                for val_dataset, val_results in all_results.items()
            },
        }

    def from_latency(key: str):
        #
        def _anonymous(res):
            if "latency_metrics" in res:
                if key in res["latency_metrics"]:
                    return res["latency_metrics"][key]

            return float("nan")

        return _anonymous

    def to_row_from_latency(key: str):
        return to_row(key, from_latency(key))

    with open(results_csv_fp, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Metric", *all_results.keys()])

        writer.writeheader()

        writer.writerow(to_row(abbrev, lambda x: x[abbrev]))

        writer.writerow(to_row_from_latency("median-emission-latency"))
        writer.writerow(to_row_from_latency("p90-emission-latency"))
        writer.writerow(to_row_from_latency("p99-emission-latency"))

        writer.writerow(to_row("EOS-fraction", lambda x: x.get("eos_frac", 0)))
        writer.writerow(to_row_from_latency("median-EOS-latency"))

        writer.writerow(to_row("SIL-fraction", lambda x: x.get("sil_frac", 0)))
        writer.writerow(to_row_from_latency("median-SIL-latency"))


if __name__ == "__main__":
    parser = val_arg_parser()
    add_val_multiple_args_in_place(parser)
    args = parser.parse_args()
    check_shared_args(args)
    check_val_multiple_args(args)
    check_val_arguments(args)

    maybe_restart_with_torchrun(
        args.num_gpus,
        args.called_by_torchrun,
        "/workspace/training/caiman_asr_train/val_multiple.py",
    )

    validate_multiple(args)
