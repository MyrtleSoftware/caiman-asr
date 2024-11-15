#!/usr/bin/env python3
import argparse
from pathlib import Path

from caiman_asr_train.utils.fast_json import fast_read_json


def get_parser():
    parser = argparse.ArgumentParser(
        "mean_json_duration.py",
        description="Calculate mean duration of utterances in JSON files",
    )
    parser.add_argument(
        "--jsons",
        type=str,
        help="Relative paths to JSON files",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Data directory containing JSON files",
        required=True,
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        help="Filter out utterances longer than this duration, default 20.0",
        default=20.0,
    )
    return parser


def get_abs_paths(json_files, data_dir):
    return [str(Path(data_dir) / f) for f in json_files]


def main(args):
    abs_paths = get_abs_paths(args.jsons, args.data_dir)
    nested_contents = (fast_read_json(path) for path in abs_paths)
    flat_contents = (item for each_json in nested_contents for item in each_json)
    durations = [
        item["original_duration"]
        for item in flat_contents
        if item["original_duration"] <= args.max_duration
    ]
    print(f"Mean duration: {sum(durations) / len(durations)}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
