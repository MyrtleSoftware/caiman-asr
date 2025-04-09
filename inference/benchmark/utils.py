import argparse
import copy
import csv
import datetime
from pathlib import Path

import boto3
from beartype import beartype
from beartype.typing import Union
from botocore import UNSIGNED
from botocore.config import Config
from ctm import CTM
from filelock import FileLock, Timeout

from caiman_asr_train.data.make_datasets.librispeech import LibriSpeech
from caiman_asr_train.data.make_datasets.librispeech import get_parser as libri_parser
from caiman_asr_train.data.make_datasets.pretty_print import pretty_path
from caiman_asr_train.utils.fast_json import fast_read_json

TMP_DIR = Path.home() / Path(".cache/myrtle/benchmark")


def shared_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--skip_transcription",
        action="store_true",
        default=False,
        help="Skip transcription step",
    )
    parser.add_argument(
        "--force_transcription",
        action="store_true",
        default=False,
        help="Force transcription even if corresponding trans files exist."
        "Otherwise, transcribe only non-existing/failed files.",
    )
    parser.add_argument(
        "--force_data_prep",
        action="store_true",
        default=False,
        help="Do the data prep step even if the data already exists",
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        default=False,
        help="Skip evaluation step",
    )
    parser.add_argument(
        "--limit_to",
        type=int,
        help="Only evaluate the first n files",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Output folder to save transcription files. "
        f"Will be interpreted relative to {pretty_path(TMP_DIR)}",
    )
    parser.add_argument(
        "--append_results",
        required=True,
        help="Results from this run will be appended to "
        "`~/.cache/myrtle/benchmark/results/[append_results].csv`",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=10,
        help="Number of times to attempt re-transcribing the same file",
    )
    parser.add_argument(
        "--custom_timestamp",
        type=str,
        default=None,
        help="Custom timestamp (instead of current time) to save in CSV",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="LibriSpeech",
        help="LibriSpeech data directory",
    )
    parser.add_argument(
        "--dset",
        type=str,
        default="librispeech-dev-clean",
        help="LibriSpeech dataset to use",
    )
    parser.add_argument(
        "--play_audio",
        action="store_true",
        help="Play audio to speakers as it's streamed. "
        "Requires aplay, incompatible with Docker",
    )

    return parser


def env_vars(args):
    TMP_DIR.mkdir(exist_ok=True, parents=True)

    data_dir = TMP_DIR / args.data_dir
    data_dir.mkdir(exist_ok=True, parents=True)
    dset = args.dset
    manifest_fpath = data_dir / f"{dset}-wav.json"
    ref_ctm_fpath = data_dir / f"{dset}.wav.ctm"

    return TMP_DIR, data_dir, dset, manifest_fpath, ref_ctm_fpath


def ftrans_is_valid(ftrans: Union[str, Path]) -> bool:
    """
    Sanity check for a single transcription file
    """
    if not Path(ftrans).is_file():
        return False

    try:
        with open(ftrans, "r") as fh:
            lines = fh.readlines()
    except UnicodeDecodeError:
        return False

    if not lines:
        return False

    for idx, line in enumerate(lines):
        entries = line.strip().split(";")
        event_type = entries[0]
        if event_type not in [
            "session_start",
            "session_end",
            "partial_received",
            "final_received",
        ]:
            return False

        if idx == 0 and not event_type == "session_start":
            return False

        if idx == len(lines) - 1 and not event_type == "session_end":
            return False

    return True


def download_source_data(dset, ref_ctm_fpath):
    dl_obj = {
        f"{dset}.wav.ctm": ref_ctm_fpath,
    }

    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    for file, dl_path in dl_obj.items():
        try:
            # Download the file
            s3_client.download_file("myrtleml-public-data", file, dl_path)
            print(f"File '{file}' downloaded successfully to '{pretty_path(dl_path)}'")
        except Exception as e:
            print(f"Error downloading file: {e}")
            raise

    src_ctm = CTM.from_file(dl_obj[f"{dset}.wav.ctm"])

    return src_ctm


@beartype
def load_manifest_or_fail(manifest_fpath: Path) -> list[dict]:
    try:
        return fast_read_json(manifest_fpath)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Manifest file not found. Try passing --force_data_prep if "
            "you want to download librispeech, or check the paths if using "
            "custom datasets."
        ) from e


@beartype
def download_librispeech_audio() -> None:
    # Download LibriSpeech
    ls_parser = libri_parser()
    ls_args = ls_parser.parse_args(
        [
            "--data_dir",
            str(TMP_DIR),
            "--convert_to_wav",
            "--dataset_parts",
            "dev-clean",
            "--force_download",
            "--use_relative_path",
        ]
    )
    librispeech = LibriSpeech(ls_args)
    librispeech.run()


@beartype
def download_librispeech_ctm(dset, ref_ctm_fpath) -> None:
    # Download ground truth CTM
    src_ctm = download_source_data(dset, ref_ctm_fpath)

    # Fix file paths in gt-CTM
    ref_ctm = CTM()
    for item in src_ctm:
        item.file = str(item.file).replace("/datasets/LibriSpeech/", "")
        ref_ctm.add_item(item)
    ref_ctm.to_file(ref_ctm_fpath)


@beartype
def maybe_download_librispeech(
    force_data_prep: bool, manifest_fpath, dset, ref_ctm_fpath
) -> None:
    lock = FileLock(TMP_DIR / "download_lock")
    try:
        lock.acquire(blocking=False)
    except Timeout:
        print(
            "Another process is already downloading the data. Waiting for it to finish..."
        )
    with lock:
        _maybe_download_librispeech(
            force_data_prep, manifest_fpath, dset, ref_ctm_fpath
        )


@beartype
def _maybe_download_librispeech(
    force_data_prep: bool, manifest_fpath, dset, ref_ctm_fpath
) -> None:
    if manifest_fpath.exists() and not force_data_prep:
        print(
            f"Skipping audio download, '{pretty_path(manifest_fpath)}' already exists."
        )
    else:
        download_librispeech_audio()
    if ref_ctm_fpath.exists() and not force_data_prep:
        print(f"Skipping ctm download, '{pretty_path(ref_ctm_fpath)}' already exists.")
    else:
        download_librispeech_ctm(dset, ref_ctm_fpath)


@beartype
def make_transcript_dir(name: str) -> Path:
    assert "," not in name, "--run_name cannot contain ','"
    path = Path(name)
    assert not path.is_absolute(), "Transcript directory must not be an absolute path"
    save_dir = TMP_DIR / path
    print(f"Transcription files will be saved to {pretty_path(save_dir)}")
    return save_dir


@beartype
def save_results_to_csv(
    wer_metrics: dict,
    latency_metrics: dict,
    utt_stats: dict,
    provider: str,
    transcript_dir: Path,
    run_name: str,
    append_results: str,
    custom_timestamp: str | None,
) -> None:
    for csv_name in [
        transcript_dir / "results.csv",
        TMP_DIR / "results" / f"{append_results}.csv",
    ]:
        csv_name.parent.mkdir(exist_ok=True, parents=True)
        write_to_csv(
            csv_name,
            wer_metrics,
            latency_metrics,
            utt_stats,
            provider,
            run_name,
            custom_timestamp,
        )


@beartype
def write_to_csv(
    csv_name: Path,
    wer_metrics: dict,
    latency_metrics: dict,
    utt_stats: dict,
    provider: str,
    run_name: str,
    custom_timestamp: str | None,
):
    exists = csv_name.exists()
    if not exists:
        print(f"Writing results to CSV file {pretty_path(csv_name)}")
    else:
        print(f"Appending results to CSV {pretty_path(csv_name)}")
    with open(csv_name, "a", newline="") as csvfile:
        fieldnames = [
            "Run name",
            "Timestamp",
            "Total utts",
            "Processed utts",
            "Mean Latency",
            "50th Percentile Latency",
            "90th Percentile Latency",
            "99th Percentile Latency",
            "WER %",
            "Total words",
            "Errors",
            "Provider",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        timestamp = {
            "Timestamp": (
                datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()
                if not custom_timestamp
                else custom_timestamp
            )
        }

        if not exists:
            writer.writeheader()
        writer.writerow(
            wer_metrics
            | fix_latency_keys(latency_metrics)
            | utt_stats
            | timestamp
            | {"Provider": provider}
            | {"Run name": run_name}
        )


def bold_yellow(text) -> str:
    return f"\033[1;33m{text}\033[0m"


def bold_green(text) -> str:
    return f"\033[1;32m{text}\033[0m"


@beartype
def fix_latency_keys(latency_metrics_original: dict) -> dict:
    """The CSV columns shouldn't change"""
    latency_metrics = copy.deepcopy(latency_metrics_original)
    latency_metrics["50th Percentile Latency"] = latency_metrics.pop(
        "median-emission-latency"
    )
    latency_metrics["Mean Latency"] = latency_metrics.pop("mean-emission-latency")
    latency_metrics["99th Percentile Latency"] = latency_metrics.pop(
        "p99-emission-latency"
    )
    latency_metrics["90th Percentile Latency"] = latency_metrics.pop(
        "p90-emission-latency"
    )
    latency_metrics.pop("stdev-emission-latency")
    return latency_metrics
