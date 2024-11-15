#!/usr/bin/env python3

import argparse
import difflib
import os
from argparse import Namespace
from dataclasses import dataclass

import matplotlib.pyplot as plt
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple

from caiman_asr_train.data.text.is_tag import is_tag
from caiman_asr_train.data.text.normalizers import lowercase_normalize as norm
from caiman_asr_train.latency.measure_latency_lite import compute_latency_metrics
from caiman_asr_train.latency.timestamp import EOS, Never, Silence, Termination
from caiman_asr_train.train_utils.distributed import print_once

BASIC_CHAR_SET = list(" abcdefghijklmnopqrstuvwxyz'")


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
    last_emit_time: Optional[Dict[str, Termination]],
    include_subs: bool = False,
) -> Tuple[List[float], List[float], List[float], List[float], float, float]:
    """
    Aligns the words from ground truth and predicted transcripts and calculates the
    emission latencies.

    Returns:
        latency: List of emission latencies.
        end_times: List of end times of ground truth words.
        sil_latency: List of end point latencies for SIL terminated utterances.
        eos_latency: List of end point latencies for EOS terminated utterances.
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
    accepted = 0
    all_gt_words = 0

    eos_latency = []
    sil_latency = []

    end_acc = 0
    end_tot = 0

    def ok(tag, d1, d2):
        if tag == "equal":
            return True

        if tag == "replace" and include_subs:
            return d1 == d2

        return False

    # Process each file separately
    for filename in ground_truth_dict:
        if filename not in predicted_dict:
            continue

        ground_truth = [
            ctm for ctm in ground_truth_dict[filename] if not is_tag(ctm.word)
        ]

        predicted = [ctm for ctm in predicted_dict[filename] if not is_tag(ctm.word)]

        ground_truth_words = [norm(ctm.word, BASIC_CHAR_SET) for ctm in ground_truth]
        predicted_words = [norm(ctm.word, BASIC_CHAR_SET) for ctm in predicted]

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
            if ok(tag, i2 - i1, j2 - j1):
                latencies.extend(
                    [
                        predicted[j].end_time - ground_truth[i].end_time
                        for i, j in zip(range(i1, i2), range(j1, j2), strict=True)
                    ]
                )
                end_times.extend(ground_truth[i].end_time for i in range(i1, i2))
                accepted += j2 - j1

        if last_emit_time is not None:
            assert filename in last_emit_time, f"Missing emit time for {filename}"

            last_gt = ground_truth_words[-1] if ground_truth_words else ""
            last_pr = predicted_words[-1] if predicted_words else ""

            if last_gt == last_pr:
                # If the ground truth CTM is empty, assume the worse possible
                # endpointing latency
                gt_final = ground_truth[-1].end_time if ground_truth else 0.0
                end_acc += 1

                match last_emit_time[filename]:
                    case EOS(final_time):
                        eos_latency.append(final_time - gt_final)
                    case Silence(final_time):
                        sil_latency.append(final_time - gt_final)
                    case Never():
                        pass

        end_tot += 1
        all_gt_words += len(ground_truth_words)

    if all_gt_words > 0:
        token_usage_rate = accepted / all_gt_words
        terminal_token_usage_rate = end_acc / end_tot
    else:
        token_usage_rate = 0.0
        terminal_token_usage_rate = 0.0
        print_once("WARNING: No ground truth words found. Please check the CTM files.")

    return (
        latencies,
        end_times,
        sil_latency,
        eos_latency,
        token_usage_rate,
        terminal_token_usage_rate,
    )


@beartype
def align_ctm_files(
    gt_ctm_paths: List[str],
    model_ctm_path: str,
    last_emit_time: Optional[Dict[str, Termination]],
    include_subs: bool = False,
) -> Tuple[List[float], List[float], List[float], List[float], float, float]:
    gt_ctm = []
    for ctm in gt_ctm_paths:
        gt_ctm += load_ctm(ctm)
    model_ctm = load_ctm(model_ctm_path)

    return align_transcripts(
        gt_ctm,
        model_ctm,
        last_emit_time,
        include_subs,
    )


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
    latencies, end_times, sil_latency, eos_latency, _, _ = align_ctm_files(
        [args.gt_ctm], args.model_ctm, None, args.include_subs
    )

    print(compute_latency_metrics(latencies, sil_latency, eos_latency))

    if args.output_img_path:
        assert (
            os.path.splitext(args.output_img_path)[1] == ".png"
        ), "Incorrect file extension for plot."
        plot_latency_vs_seq_len(latencies, end_times, args.output_img_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
