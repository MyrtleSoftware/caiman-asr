import csv
import json
import os
import tarfile
from argparse import Namespace
from pathlib import Path

import torch.distributed as dist
from beartype import beartype
from beartype.typing import Dict, List, Optional, Union

from caiman_asr_train.latency.measure_latency import (
    align_ctm_files,
    compute_latency_metrics,
)
from caiman_asr_train.latency.timestamp import SequenceTimestamp
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
        for word_align in seq_time.seqs:
            start_time = word_align.start_frame * frame_width  # convert to seconds
            duration = (word_align.end_frame - word_align.start_frame) * frame_width
            # For single channel audio, channel_id is always 1
            file.write(
                f"{audio_fp} 1 {start_time:.3f} {duration:.3f} {word_align.word} \n"
            )


@beartype
def manage_ctm_export(
    args: Namespace,
    lst_seq_time: List[SequenceTimestamp],
    gt_ctm_fpaths: List[str],
    flist: Union[List[str], None],
) -> Dict[str, Optional[float]]:
    """
    Manages the export of word-level timestamps to a CTM file, accommodating different
    data sources and computing environments.

    In a multi-GPU setup, it ensures that only the process running on GPU:0 performs
    the CTM export to avoid duplication.

    The function processes each audio file, exports the corresponding CTM file, and
    calculates emission latencies by aligning these exported CTM files with the provided
    ground truth CTM files.
    """
    latency_metrics = {}
    if args.num_gpus == 1 or dist.get_rank() == 0:
        frame_width = encoder_output_frame_width(args.model_config)
        index = 0
        if args.read_from_tar:
            data = get_audio_filenames_from_tar(args.dataset_dir, args.val_tar_files)
            file_list = args.val_tar_files
        else:
            data = combine_manifest_data(args.dataset_dir, args.val_manifests)
            file_list = args.val_manifests

        data = [j for i in data for j in i]
        base_name = "-".join([Path(x).stem for x in file_list])
        ctm_fpath = str(Path(args.output_dir) / f"{base_name}_{args.timestamp}.ctm")
        for i, sample in enumerate(data):
            to_ctm(
                lst_seq_time[index],
                ctm_fpath,
                sample if flist is None else flist[i],
                frame_width,
            )
            index += 1
        latencies, _ = align_ctm_files(gt_ctm_fpaths, ctm_fpath)
        latency_metrics = compute_latency_metrics(latencies)

    return latency_metrics


@beartype
def manage_multiple_ctm_export(
    args: Namespace,
    all_timestamps: dict,
    gt_ctm_fpaths: List[str],
    multi_gpu: bool = False,
) -> None:
    """
    Manages the export of word-level timestamps to a CTM file, for val_multiple.

    The function processes each audio file, exports the corresponding CTM file, and
    calculates emission latencies by aligning these exported CTM files with the
    provided ground truth CTM files. The latency metrics are printed and saved to csv
    and json.
    """
    if not multi_gpu or dist.get_rank() == 0:
        frame_width = encoder_output_frame_width(args.model_config)
        all_latencies = {}
        for i, (manifest_path, timestamps) in enumerate(all_timestamps.items()):
            data = combine_manifest_data(args.all_dataset_dirs[i], [manifest_path])[0]
            base_name = os.path.splitext(os.path.basename(manifest_path))[0]
            output_ctm_fpath = os.path.join(
                args.output_dir, f"{base_name}_{args.timestamp}.ctm"
            )
            for j, audio_filename in enumerate(data):
                to_ctm(
                    timestamps[j],
                    output_ctm_fpath,
                    audio_filename,
                    frame_width,
                )
            latencies, _ = align_ctm_files([gt_ctm_fpaths[i]], output_ctm_fpath)
            print(f"Emission latencies for {manifest_path}")
            all_latencies[manifest_path] = compute_latency_metrics(latencies)

        csv_path = os.path.join(args.output_dir, "validate_multiple.csv")
        json_path = os.path.join(args.output_dir, "validate_multiple.json")

        # csv formatting:
        # dataset1, dataset2, ..., dataset1 EL, dataset2 EL, ...
        # 8.93, 9.20, ..., 0.332, 0.342, ...
        # with WER columns first, followed by emission latency columns
        # csv only contains mean latencies (in seconds)

        with open(csv_path, "r") as file:
            reader = csv.reader(file)
            original_data = list(reader)

        headers = original_data[0]
        data_row = original_data[1]

        for dataset in all_latencies.keys():
            headers.append(f"{dataset} mean EL")
            data_row.append(round(all_latencies[dataset]["Mean Latency"], 4))

        with open(csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows([data_row])

        # json contains mean and percentile (50th and 90th) latencies

        with open(json_path, "r") as file:
            data = json.load(file)

        data["latencies"] = all_latencies

        with open(json_path, "w") as file:
            json.dump(data, file, indent=4)


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
        assert os.path.exists(
            ctm_file
        ), f"Corresponding ground truth CTM file {ctm_file} not found"
        ctms_files.append(ctm_file)

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
def combine_manifest_data(data_dir: str, val_manifests: List[str]) -> List[List[str]]:
    """Combine JSON data from multiple manifest files.

    Parameters:
    data_dir (str): The directory path where data is stored.
    val_manifests (List[str]): A list of file paths for validation manifests.

    Returns:
    List[List[str]]:  A list of lists of audio filenames for each manifest file.
    """
    abs_manifest_paths = get_abs_manifest_paths(data_dir, val_manifests)
    audio_filenames_per_file = []

    for manifest_path in abs_manifest_paths:
        audio_filenames = []
        with open(manifest_path, "r") as file:
            data = json.load(file)
            for sample in data:
                audio_filenames.append(sample["files"][0]["fname"])
        audio_filenames_per_file.append(audio_filenames)
    return audio_filenames_per_file


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
