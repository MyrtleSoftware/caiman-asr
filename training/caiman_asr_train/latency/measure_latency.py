#!/usr/bin/env python3

import argparse
import difflib
import os
from argparse import Namespace
from dataclasses import dataclass

import matplotlib.pyplot as plt
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple


@dataclass
class CTMTimestamp:
    word: str
    end_time: float
    filename: str


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_ctm",
        required=True,
        type=str,
        help="Absolute path to ground truth ctm file",
    )
    parser.add_argument(
        "--model_ctm", required=True, type=str, help="Absolute path to model ctm file"
    )
    parser.add_argument(
        "--include_subs",
        action="store_true",
        default=False,
        help="Include substitution errors in latency computation",
    )
    parser.add_argument(
        "--output_img_path",
        default=None,
        type=str,
        help="Absolute output path for latency vs sequence length graph",
    )
    return parser.parse_args()


@beartype
def load_ctm(ctm_file_path: str) -> List[CTMTimestamp]:
    """
    Loads a CTM file & parses into a list of tuples containing word and its end times.

    Args:
        ctm_file_path: Path to the CTM file.

    Returns:
        ctm_data: List of CTMTimestamp dataclass objects.

    Raises:
        ValueError: If a line in the CTM file does not conform to the expected format.
        IOError: If there is an error reading the CTM file.
    """
    ctm_data = []
    try:
        with open(ctm_file_path, "r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 5:
                    raise ValueError(
                        f"Incorrect CTM format in file {ctm_file_path} on line: "
                        f"{line.strip()}"
                    )
                filename, _, start_time, duration, word = parts[:5]
                ctm_data.append(
                    CTMTimestamp(word, float(start_time) + float(duration), filename)
                )
    except IOError as e:
        print(f"Error reading CTM file: {ctm_file_path}\n{e}")
        raise
    return ctm_data


@beartype
def align_transcripts(
    ground_truth: List[CTMTimestamp],
    predicted: List[CTMTimestamp],
    include_subs: bool = False,
) -> Tuple[List[float], List[float]]:
    """
    Aligns the words from ground truth and predicted transcripts and calculates the
    emission latencies.
    """
    # Create dictionaries to group words by filenames
    ground_truth_dict = {}
    predicted_dict = {}

    for ctm_timestamp in ground_truth:
        ground_truth_dict.setdefault(ctm_timestamp.filename, []).append(ctm_timestamp)

    for ctm_timestamp in predicted:
        predicted_dict.setdefault(ctm_timestamp.filename, []).append(ctm_timestamp)

    latencies = []
    end_times = []

    # Process each file separately
    for filename in ground_truth_dict:
        if filename not in predicted_dict:
            continue

        ground_truth_words = [ctm.word for ctm in ground_truth_dict[filename]]
        predicted_words = [ctm.word for ctm in predicted_dict[filename]]

        # Initialize the SequenceMatcher for each file
        matcher = difflib.SequenceMatcher(
            None, ground_truth_words, predicted_words, autojunk=False
        )

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            # opcodes are instructions ('equal', 'replace', 'delete', 'insert')
            # for transforming ground_truth into predicted.
            # Each opcode (tag, i1, i2, j1, j2) specifies the operation and
            # start/end indices in both sequences.

            # Process according to the tag and include_subs flag
            if tag in ("equal", "replace") and (tag != "replace" or include_subs):
                latencies.extend(
                    [
                        predicted_dict[filename][j].end_time
                        - ground_truth_dict[filename][i].end_time
                        for i, j in zip(range(i1, i2), range(j1, j2), strict=True)
                    ]
                )
                end_times.extend(
                    ground_truth_dict[filename][i].end_time for i in range(i1, i2)
                )

    return latencies, end_times


@beartype
def remove_outliers(
    latencies, timestamps, threshold=2
) -> Tuple[List[float], List[float]]:
    """
    Remove outlier latencies outside threshold and their corresponding timestamps.

    Args:
    -----
    latencies: List of latency measurements.
    timestamps: List of timestamps corresponding to each latency measurement.
    threshold: The threshold range to determine outliers (default is 2).

    Returns:
    --------
    filtered_latencies: The list of latencies with outliers removed.
    filtered_timestamps: The list of timestamps corresponding to the filtered latencies.
    """
    filtered_latencies = []
    filtered_timestamps = []

    for latency, timestamp in zip(latencies, timestamps):
        if -threshold <= latency <= threshold:
            filtered_latencies.append(latency)
            filtered_timestamps.append(timestamp)

    return filtered_latencies, filtered_timestamps


@beartype
def align_ctm_files(
    gt_ctm_paths: List[str], model_ctm_path: str, include_subs: bool = False
) -> Tuple[List[float], List[float]]:
    gt_ctm = []
    for ctm in gt_ctm_paths:
        gt_ctm += load_ctm(ctm)
    model_ctm = load_ctm(model_ctm_path)
    latencies, end_times = align_transcripts(gt_ctm, model_ctm, include_subs)
    latencies, end_times = remove_outliers(latencies, end_times)
    return latencies, end_times


@beartype
def compute_latency_metrics(
    latencies: List[float],
    percentiles: List[float] = [50, 90],
) -> Dict[str, Optional[float]]:
    keys = ["Mean Latency"] + [f"{i}th Percentile" for i in percentiles]
    latency_metrics = {key: None for key in keys}
    latency_num = len(latencies)

    if not latency_num:
        return latency_metrics

    latencies = sorted(latencies)
    latency_metrics["Mean Latency"] = round(sum(latencies) / latency_num, 3)
    for perc in percentiles:
        latency_percentile = latencies[int(latency_num * perc / 100)]
        latency_metrics[f"{perc}th Percentile"] = round(latency_percentile, 3)

    return latency_metrics


@beartype
def plot_latency_vs_seq_len(latencies, end_times, save_path):
    """
    Plots emission latency as a function of timestamps and saves the plot to a file.

    Args:
        latencies (List[float]): A list of emission latencies.
        end_times (List[float]): A list of timestamps in seconds.
        save_path (str): File path where the plot will be saved.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(end_times, latencies, alpha=0.2)
    plt.xlabel("Time from start of sequence (seconds)")
    plt.ylabel("Emission Latency (seconds)")
    plt.title("Emission Latency vs. Sequence Length")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def main(args: Namespace):
    latencies, end_times = align_ctm_files(
        [args.gt_ctm], args.model_ctm, args.include_subs
    )
    print(compute_latency_metrics(latencies))
    if args.output_img_path:
        assert (
            os.path.splitext(args.output_img_path)[1] == ".png"
        ), "Incorrect file extension for plot."
        plot_latency_vs_seq_len(latencies, end_times, args.output_img_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
