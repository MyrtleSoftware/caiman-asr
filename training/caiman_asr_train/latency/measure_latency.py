#!/usr/bin/env python3

import argparse
import difflib
import os
from argparse import Namespace
from dataclasses import dataclass
from statistics import mean, median

import matplotlib.pyplot as plt
from beartype import beartype
from beartype.typing import Any, Callable, Dict, List, Optional, Tuple
from lhotse import CutSet

from caiman_asr_train.data.text.allowed_charset import get_allowed_charset
from caiman_asr_train.data.text.is_tag import is_tag
from caiman_asr_train.data.text.normalizers import lowercase_normalize as norm
from caiman_asr_train.latency.measure_latency_lite import compute_latency_metrics
from caiman_asr_train.latency.timestamp import EOS, Never, Silence, Termination
from caiman_asr_train.train_utils.distributed import print_once


@dataclass
class CTMTimestamp:
    word: str
    beg_time: float
    end_time: float
    filename: str


@beartype
@dataclass
class Latencies:
    raw_latency: List[float]
    fw_latency: List[float]
    sil_latency: List[float]
    eos_latency: List[float]


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
        "--lang_code",
        type=str,
        required=True,
        help="ISO language code to use for normalizing transcripts",
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
    parser.add_argument(
        "--frame_width",
        default=0.0,
        type=float,
        help=(
            "The expected frame latency is computed from this and "
            "sutracted from the emission latency statistics"
        ),
    )
    return parser.parse_args()


@beartype
def load_ctm_from_cuts(manifest_path: str) -> list[Any]:
    cutset = CutSet.from_jsonl(manifest_path)
    ctm_data = []
    for cut in cutset:
        filename = cut.recording.sources[0].source
        for sup in cut.supervisions:
            if sup.alignment is None or "word" not in sup.alignment:
                continue

            for word, start_time, duration, _ in sup.alignment["word"]:
                ctm_data.append(
                    CTMTimestamp(
                        word,
                        float(start_time),
                        float(start_time) + float(duration),
                        filename,
                    )
                )

    if not ctm_data:
        print_once(f"WARNING: CTM not found for manifest {manifest_path}.")

    return ctm_data


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
                    CTMTimestamp(
                        word,
                        float(start_time),
                        float(start_time) + float(duration),
                        filename,
                    )
                )
    except IOError as e:
        print(f"Error reading CTM file: {ctm_file_path}\n{e}")
        raise
    return ctm_data


class _Delta:
    def __init__(self, gt_beg, gt_end, pr_beg, pr_end):
        # gt = ground truth
        # pr = predicted
        self.gt_end = gt_end

        self.head_lat = pr_beg - gt_beg
        self.tail_lat = pr_end - gt_end

        self.time_gt = gt_end - gt_beg
        self.time_pr = pr_end - pr_beg


@beartype
def timestamp_stats(
    deltas: list[_Delta],
    head_offset: float,
    tail_offset: float,
) -> Dict[str, float]:
    if not deltas:
        return {}

    @beartype
    def correct(off: float) -> Callable[[float], float]:
        return lambda x: abs(x - off)

    mean_word_gt = mean(x.time_gt for x in deltas)
    mean_word_pr = mean(x.time_pr for x in deltas)

    mean_raw_head = mean(map(correct(0.0), (x.head_lat for x in deltas)))
    mean_raw_tail = mean(map(correct(0.0), (x.tail_lat for x in deltas)))
    raw_AAS = (mean_raw_head + mean_raw_tail) / 2

    mean_head = mean(map(correct(head_offset), (x.head_lat for x in deltas)))
    mean_tail = mean(map(correct(tail_offset), (x.tail_lat for x in deltas)))
    fix_AAS = (mean_head + mean_tail) / 2

    optimal_head_offset = median(x.head_lat for x in deltas)
    optimal_tail_offset = median(x.tail_lat for x in deltas)
    c_mean_beg = mean(map(correct(optimal_head_offset), (x.head_lat for x in deltas)))
    c_mean_end = mean(map(correct(optimal_tail_offset), (x.tail_lat for x in deltas)))
    corrected_AAS = (c_mean_beg + c_mean_end) / 2

    timestamp_stats = {
        "mean_word_time_gt": mean_word_gt,
        "mean_word_time_pr": mean_word_pr,
        "optimal_head_offset": optimal_head_offset,
        "optimal_tail_offset": optimal_tail_offset,
        "raw_AAS": raw_AAS,
        "fixed_AAS": fix_AAS,
        "corrected_AAS": corrected_AAS,
    }

    return timestamp_stats


@beartype
def align_transcripts(
    ground_truth: List[CTMTimestamp],
    predicted: List[CTMTimestamp],
    last_emit_time: Optional[Dict[str, Termination]],
    head_offset: float,
    tail_offset: float,
    lang_code: str,
    include_subs: bool = False,
) -> Tuple[Latencies, List[float], float, float, dict[str, float]]:
    """
    Aligns the words from ground truth and predicted transcripts and calculates the
    emission latencies.

    Returns:
        latencies: Latencies dataclass object containing:
            - raw_latency: List of emission latencies.
            - fw_latency: List of latencies for the first matched word in each file.
            - sil_latency: List of end point latencies for SIL terminated utterances.
            - eos_latency: List of end point latencies for EOS terminated utterances.
        end_times: List of end times of ground truth words.
        token_usage_rate: Fraction of ground truth words successfully matched (Recall).
        terminal_token_usage_rate: %files with a correctly matched terminal token.
        t_stats: Dictionary of alignment metrics, offsets, and Average Alignment Scores.
    """
    # Create dictionaries to group words by filenames
    ground_truth_dict = {}
    predicted_dict = {}

    for ctm_timestamp in ground_truth:
        ground_truth_dict.setdefault(ctm_timestamp.filename, []).append(ctm_timestamp)

    for ctm_timestamp in predicted:
        predicted_dict.setdefault(ctm_timestamp.filename, []).append(ctm_timestamp)

    accepted = 0
    all_gt_words = 0

    deltas = []
    fw_latency = []
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

    charset = get_allowed_charset([lang_code], lowercase_only=True)

    # Process each file separately
    for filename in ground_truth_dict:
        if filename not in predicted_dict:
            continue

        ground_truth = [
            ctm for ctm in ground_truth_dict[filename] if not is_tag(ctm.word)
        ]

        predicted = [ctm for ctm in predicted_dict[filename] if not is_tag(ctm.word)]

        ground_truth_words = [
            norm(ctm.word, charset, lang_code) for ctm in ground_truth
        ]
        predicted_words = [norm(ctm.word, charset, lang_code) for ctm in predicted]

        # Initialize the SequenceMatcher for each file
        matcher = difflib.SequenceMatcher(
            None, ground_truth_words, predicted_words, autojunk=False
        )

        if (
            ground_truth_words
            and predicted_words
            and ground_truth_words[0] == predicted_words[0]
        ):
            fw_latency.append(predicted[0].end_time - ground_truth[0].end_time)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            # opcodes are instructions ('equal', 'replace', 'delete', 'insert')
            # for transforming ground_truth into predicted.
            # Each opcode (tag, i1, i2, j1, j2) specifies the operation and
            # start/end indices in both sequences.

            # Process according to the tag and include_subs flag
            if ok(tag, i2 - i1, j2 - j1):
                deltas.extend(
                    _Delta(
                        ground_truth[i].beg_time,
                        ground_truth[i].end_time,
                        predicted[j].beg_time,
                        predicted[j].end_time,
                    )
                    for i, j in zip(range(i1, i2), range(j1, j2), strict=True)
                )
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

    t_stats = timestamp_stats(deltas, head_offset=head_offset, tail_offset=tail_offset)

    if all_gt_words > 0:
        token_usage_rate = accepted / all_gt_words
        terminal_token_usage_rate = end_acc / end_tot
    else:
        token_usage_rate = 0.0
        terminal_token_usage_rate = 0.0
        print_once("WARNING: No ground truth words found. Please check the CTM files.")

    latencies = Latencies(
        [x.tail_lat for x in deltas],
        fw_latency,
        sil_latency,
        eos_latency,
    )

    return (
        latencies,
        [x.gt_end for x in deltas],
        token_usage_rate,
        terminal_token_usage_rate,
        t_stats,
    )


@beartype
def align_ctm_files(
    gt_ctm_paths: List[str],
    model_ctm_path: str,
    last_emit_time: Optional[Dict[str, Termination]],
    lang_code: str,
    include_subs: bool = False,
    head_offset: float = 0.0,
    tail_offset: float = 0.0,
) -> Tuple[Latencies, List[float], float, float, dict[str, float]]:
    gt_ctm = []
    for ctm in gt_ctm_paths:
        gt_ctm += load_ctm_from_cuts(ctm)
    model_ctm = load_ctm(model_ctm_path)

    return align_transcripts(
        gt_ctm,
        model_ctm,
        last_emit_time,
        include_subs=include_subs,
        head_offset=head_offset,
        tail_offset=tail_offset,
        lang_code=lang_code,
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
    latencies, end_times, *_ = align_ctm_files(
        [args.gt_ctm], args.model_ctm, None, args.lang_code, args.include_subs
    )

    print(
        compute_latency_metrics(
            latencies=latencies.raw_latency,
            fw_latency=latencies.fw_latency,
            sil_latency=latencies.sil_latency,
            eos_latency=latencies.eos_latency,
            frame_width=args.frame_width,
        )
    )

    if args.output_img_path:
        assert (
            os.path.splitext(args.output_img_path)[1] == ".png"
        ), "Incorrect file extension for plot."
        plot_latency_vs_seq_len(latencies.raw_latency, end_times, args.output_img_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
