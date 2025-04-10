import os
import tarfile
from argparse import Namespace
from pathlib import Path

import torch.distributed as dist
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple

from caiman_asr_train.latency.measure_latency import align_ctm_files
from caiman_asr_train.latency.measure_latency_lite import compute_latency_metrics
from caiman_asr_train.latency.timestamp import (
    SequenceTimestamp,
    Termination,
    frame_to_time,
)
from caiman_asr_train.train_utils.distributed import print_once
from caiman_asr_train.utils.frame_width import encoder_output_frame_width


@beartype
def to_ctm(
    seq_time: SequenceTimestamp, output_fp: str, audio_fp: str, frame_width: float
) -> None:
    """Appends word-level timestamps from a SequenceTimestamp object to a CTM file.

    CTM format is space separated file with entries:
    <recording_id> <channel_id> <token_start_ts> <token_duration_ts> <token_value>
    and an optional sixth entry <confidence_score>.
    CTM allows either token-level or word-level timestamps.

    Parameters
    ----------
    seq_time
        A SequenceTimestamp object containing per-word timestamps within a sentence.
    output_fp
        File path to the CTM file where the output will be appended.
    audio_fp
        The file path of the audio file corresponding to the SequenceTimestamp data.
    frame_width
        Duration of frame width in seconds.
    """
    with open(output_fp, "a") as file:
        # Iterate over each word in sentence
        for word_align in map(lambda x: frame_to_time(x, frame_width), seq_time.seqs):
            start_time = word_align.start_time
            duration = word_align.end_time - word_align.start_time

            # For single channel audio, channel_id is always 1
            file.write(
                f"{audio_fp} 1 {start_time:.3f} {duration:.3f} {word_align.word} \n"
            )


@beartype
def dump_ctm(
    flist: List[str],
    lst_seq_time: List[SequenceTimestamp],
    ctm_fpath: str,
    frame_width: float,
) -> Dict[str, Termination]:
    for time, sample in zip(lst_seq_time, flist, strict=True):
        to_ctm(time, ctm_fpath, sample, frame_width)

    return {fname: seq.eos for fname, seq in zip(flist, lst_seq_time, strict=True)}


@beartype
def manage_ctm_export(
    args: Namespace,
    lst_seq_time: List[SequenceTimestamp],
    gt_ctm_fpaths: List[str],
    flist: List[str],
) -> Tuple[
    Dict[str, Optional[float]], dict[str, float], List[float], List[float], List[float]
]:
    """
    Manages the export of word-level timestamps to a CTM file, accommodating different
    data sources and computing environments.

    In a multi-GPU setup, it ensures that only the process running on GPU:0 performs
    the CTM export to avoid duplication.

    The function processes each audio file, exports the corresponding CTM file, and
    calculates emission latencies by aligning these exported CTM files with the provided
    ground truth CTM files.

    Returns:
        latency_metrics: A dictionary containing latency metrics.
        latencies: A list of emission latencies.
        sil_latency: A list of silence latencies.
        eos_latency: A list of end-of-sentence latencies.
    """
    latency_metrics = {}
    timestamp_metrics = {}
    latencies = []
    sil_latency = []
    eos_latency = []

    if args.num_gpus == 1 or dist.get_rank() == 0:
        frame_width = encoder_output_frame_width(args.model_config)

        if args.read_from_tar:
            file_list = args.val_tar_files
        else:
            file_list = args.val_manifests

        base_name = "-".join([Path(x).stem for x in file_list])

        ctm_fpath = str(Path(args.output_dir) / f"{base_name}_{args.timestamp}.ctm")

        # Make function repeatable by clearing the file
        with open(ctm_fpath, "w") as _:
            pass

        last_emit_time = dump_ctm(flist, lst_seq_time, ctm_fpath, frame_width)

        (
            latencies,
            _,
            sil_latency,
            eos_latency,
            token_usage_rate,
            terminal_token_usage_rate,
            timestamp_metrics,
        ) = align_ctm_files(
            gt_ctm_fpaths,
            ctm_fpath,
            last_emit_time,
            head_offset=args.latency_head_offset,
            tail_offset=args.latency_tail_offset,
        )

        if token_usage_rate <= 0.50:
            print(f"WARNING: {token_usage_rate=} is very low (below 50%).")

        if terminal_token_usage_rate <= 0.50:
            print(f"WARNING: {terminal_token_usage_rate=} is very low (below 50%).")

        latency_metrics = compute_latency_metrics(
            latencies, sil_latency, eos_latency, frame_width=frame_width
        )

        latency_metrics["token_usage_rate"] = token_usage_rate
        latency_metrics["terminal_token_usage_rate"] = terminal_token_usage_rate

    return latency_metrics, timestamp_metrics, latencies, sil_latency, eos_latency


@beartype
def get_reference_ctms(args: Namespace, val_multiple: bool = False) -> List[str]:
    """
    Get reference ctms for validation manifests.
    """
    if val_multiple:
        manifests_files = []
        for manifest, dataset_dir in zip(args.all_val_manifests, args.all_dataset_dirs):
            manifests_files.append(get_abs_path(dataset_dir, manifest))
    elif args.read_from_tar:
        manifests_files = get_abs_tar_paths(args.dataset_dir, args.val_tar_files)
    else:
        manifests_files = get_abs_manifest_paths(args.dataset_dir, args.val_manifests)

    ctms_files = []
    for manifest_file in manifests_files:
        ctm_file = os.path.splitext(manifest_file)[0] + ".ctm"

        if os.path.exists(ctm_file):
            ctms_files.append(ctm_file)
        else:
            print_once(f"WARNING: CTM file for {manifest_file} not found")

    return ctms_files


@beartype
def get_abs_manifest_paths(data_dir: str, val_manifests: List[str]) -> List[str]:
    """Gets the absolute paths of the validation manifests files.

    Manifest file paths are unsorted.

    Parameters:
    data_dir (str): The directory path where data is stored.
    val_manifests (List[str]): A list of file paths for validation manifests. Each path
                               can be either an absolute path or a relative path.

    Returns:
    List[str]: A list of absolute paths to the validation manifests files.
    """
    abs_paths = []
    for manifest in val_manifests:
        abs_paths.append(get_abs_path(data_dir, manifest))
    return abs_paths


@beartype
def get_audio_filenames_from_tar(
    data_dir: str, tar_paths: List[str]
) -> List[List[str]]:
    """Extracts audio filenames (.flac or .wav) from tar files, ignoring .txt files.

    Parameters:
    data_dir (str): The directory path where data is stored.
    tar_path (List[str]): List of path to tar files.

    Returns:
    List[List[str]]: A list of lists of audio filenames in the order they appear for
    each tar file.
    """
    abs_tar_paths = get_abs_tar_paths(data_dir, tar_paths)
    audio_filenames_per_tar = []

    for tar_path in abs_tar_paths:
        audio_filenames = []
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith((".flac", ".wav")):
                    audio_filenames.append(member.name)
        audio_filenames_per_tar.append(audio_filenames)
    return audio_filenames_per_tar


@beartype
def get_abs_tar_paths(data_dir: str, tar_paths: List[str]) -> List[str]:
    """Gets the absolute paths of tar files.

    Tar filepaths are sorted alphabetically.

    Parameters:
    data_dir (str): The directory path where data is stored.
    tar_paths (List[str]):  A list of file paths for tar files, which can be
                            either absolute or relative.

    Returns:
    List[str]: A list of absolute paths to the tar files.
    """
    abs_paths = []
    for path in sorted(tar_paths):
        abs_paths.append(get_abs_path(data_dir, path))
    return abs_paths


@beartype
def get_abs_path(directory: str, path: str) -> str:
    """Returns abs path of path, relative to directory if not already absolute."""
    if os.path.isabs(path):
        return path
    else:
        return os.path.join(directory, path)
