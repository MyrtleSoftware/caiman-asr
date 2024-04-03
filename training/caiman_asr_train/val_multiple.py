#!/usr/bin/env python3
import csv
import json
from argparse import ArgumentParser, Namespace
from copy import copy
from pathlib import Path

from caiman_asr_train.args.val import check_val_arguments
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
        parser.error(
            f"'--all_val_manifests'={args.all_val_manifests} and "
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
            f"Attempting to overwrite {results_fp}. Either pass a new --results_fp "
            "or, if overwriting is the intended behaviour, pass --overwrite_ok"
        )
    results_fp.parent.mkdir(exist_ok=True)

    setup_class = ValCPUSetup if args.cpu else ValSetup
    results = {}
    all_timestamps = {}
    skip_init = False
    for manifest, dataset_dir in zip(args.all_val_manifests, args.all_dataset_dirs):
        val_name = Path(manifest).with_suffix("").name
        output_dir = Path(args.output_dir) / val_name
        output_dir.mkdir(exist_ok=True)

        val_args = copy(args)
        val_args.val_manifests = [manifest]
        val_args.dataset_dir = dataset_dir
        val_args.output_dir = str(output_dir)
        val_args.skip_init = skip_init

        val_objects = setup_class().run(val_args)
        model_results = validate(val_args, val_objects=val_objects)
        wer = round(model_results["wer"], 5)
        manifest_path = Path(val_args.dataset_dir) / manifest
        results[str(manifest_path)] = wer
        all_timestamps[str(manifest_path)] = model_results["timestamps"]

        skip_init = True  # only initialise logging once

    out_json = {"wer": results, "args": args.__dict__}
    with results_fp.open("w") as f:
        json.dump(out_json, f)

    # Write to csv
    with open(results_csv_fp, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)

    return all_timestamps


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
