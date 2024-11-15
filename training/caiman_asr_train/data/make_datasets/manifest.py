#!/usr/bin/env python
# Copyright (c) 2024, Myrtle.ai. All rights reserved.

import logging
import multiprocessing
from functools import partial
from pathlib import Path

import pandas as pd
import sox
from beartype import beartype
from beartype.typing import Dict, Iterable, List, Union
from tqdm import tqdm

from caiman_asr_train.data.make_datasets.pretty_print import pretty_path
from caiman_asr_train.utils.fast_json import fast_write_json


def save_manifest(manifest: List[Dict], fpath: Union[str, Path]) -> None:
    logging.info(
        f"Saving {pretty_path(Path(fpath))} manifest to disk, "
        f"contains {len(manifest)} entries"
    )
    fast_write_json(manifest, fpath)


def _validate(item: Dict, data_dir: Union[str, Path]):
    """
    A single-thread validation routine. Expected to be called by `validate_manifest`
    multiprocessing pool.
    """
    fpath = data_dir / item["files"][0]["fname"]
    file_info = sox.file_info.info(str(fpath))

    # Check audio file exists
    if not Path(fpath).is_file():
        return False, f"file {fpath} does not exist"

    # Check transcript is not empty
    if not item["transcript"]:
        return False, f"{fpath} transcript is empty"

    # Check relevant audio metadata
    if not item["original_duration"] == file_info["duration"]:
        return False, f"{fpath} faulty duration"

    if not item["original_num_samples"] == file_info["num_samples"]:
        return False, f"{fpath} faulty number of samples"

    return True, ""


def validate_manifest(
    manifest: List[Dict], data_dir: Union[str, Path, None] = None, num_jobs: int = 8
):
    """
    Validate manifest:
        1) all audio files exist
        2) no transcript is empty
        3) relevant audio metadata is correct
        4) there are no duplicate audio files
    """
    if data_dir is None:
        data_dir = Path("/")
    else:
        data_dir = Path(data_dir)

    with multiprocessing.Pool(num_jobs) as pool:
        msgs = pool.starmap(_validate, [(item, data_dir) for item in manifest])

    for error, msg in msgs:
        assert error, f"{msg}"


@beartype
def process_utterance(
    data_dir: Union[str, Path, None], use_relative_path: bool, data: Dict
) -> Dict:
    """
    Process single data item (dict) which is of following format:
    {
        'audio_file': name of audio with extension,
        'transcript': transcript as string,
    }

    The output format is:
    {
        "transcript": "BLA BLA BLA ...",
        "files": [
            {
            "channels": 1,
            "sample_rate": 16000.0,
            "bitdepth": 16,
            "bitrate": 155000.0,
            "duration": 11.21,
            "num_samples": 179360,
            "encoding": "FLAC",
            "silent": false,
            "fname": "test-clean/5683/32879/5683-32879-0004.flac"
            }
        ],
        "original_duration": 11.21,
        "original_num_samples": 179360
    }
    Args:
    :data: a dictionary describing a single utterance
    :data_dir: a data directory path where to search for audio files, needed if
        the manifest contains relative paths
    """
    if data_dir is None:
        data_dir = Path("/")
    else:
        data_dir = Path(data_dir)
    audio_file = data["audio_file"]
    trans = data["transcript"]
    abs_audio_file = data_dir / Path(audio_file)

    file_info = sox.file_info.info(str(abs_audio_file))
    file_info["fname"] = str(audio_file) if use_relative_path else str(abs_audio_file)

    return {
        "transcript": trans,
        "files": [file_info],
        "original_duration": file_info["duration"],
        "original_num_samples": file_info["num_samples"],
    }


@beartype
def prepare_manifest(
    data: Iterable[Dict],
    num_jobs: int = 1,
    data_dir: Union[str, Path, None] = None,
    use_relative_path: bool = False,
) -> List[Dict]:
    """
    Takes in `data` and creates a manifest in a parallel fashion. The `data_dir`
    argument is needed if the `audio_file` is relative and not absolute. The output
    manifest will contain audio filepaths based on this setting.

    Data is in the following format:
    [
        {
            'audio_file': name of audio with extension,
            'transcript': transcript as string,
        }
    ]

    Manifest is a list of dictionaries, i.e.:
    [
        {
        "transcript": "BLA BLA BLA ...",
        "files": [
            {
            "channels": 1,
            "sample_rate": 16000.0,
            "bitdepth": 16,
            "bitrate": 155000.0,
            "duration": 11.21,
            "num_samples": 179360,
            "encoding": "FLAC",
            "silent": false,
            "fname": "test-clean/5683/32879/5683-32879-0004.flac"
            }
        ],
        "original_duration": 11.21,
        "original_num_samples": 179360
        },
        ...
    ]

    Args:
    :data: a list of items where each item contains info on an utterance
    :num_jobs: a number of jobs to run the process in parallel
    :data_dir: a data directory path where to search for audio files, needed if
        the manifest contains relative paths
    """
    # run multiprocessing over all items in data
    process_utterance_ = partial(process_utterance, data_dir, use_relative_path)

    try:
        data_len = len(data)
    except TypeError:
        data_len = None
    with multiprocessing.Pool(num_jobs) as pool:
        dataset = list(tqdm(pool.imap(process_utterance_, data), total=data_len))
    dataset = sorted(dataset, key=lambda x: x["files"][0]["fname"])
    dataset = list(filter(all_fields_exist, dataset))

    # transform into -> pandas dataframe -> JSON
    df = pd.DataFrame(dataset, dtype=object)
    manifest = df.to_dict(orient="records")

    return manifest


def all_fields_exist(utterance: Dict) -> bool:
    any_none = any(v is None for v in utterance.values()) or any(
        v is None for v in utterance["files"][0].values()
    )
    if any_none:
        print(f"Warning: Filtering out {utterance}")
    return not any_none
