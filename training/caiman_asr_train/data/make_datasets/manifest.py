#!/usr/bin/env python
# Copyright (c) 2024, Myrtle.ai. All rights reserved.

import json
import logging
import multiprocessing
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import sox


def load_manifest(fpath: Union[str, Path]) -> List[Dict]:
    with open(fpath, "r") as fp:
        manifest = json.load(fp)
    return manifest


def save_manifest(manifest: List[Dict], fpath: Union[str, Path]) -> None:
    logging.info(
        f"Saving {fpath} manifest to disk, " f"contains {len(manifest)} entries"
    )
    with open(fpath, "w") as fp:
        json.dump(manifest, fp, indent=2)


def validate_manifest(manifest: List[Dict], data_dir: Union[str, Path, None] = None):
    """
    Validate a manifest which is a list of dictionaries. The `data_dir` argument
    is required if audio filepaths are relative and not absolute. Otherwise, leave
    it as None. The validation process checks the following:
        1) all audio files exist
        2) no transcript is empty
        3) relevant audio metadata is correct
        4) there are no duplicate audio files

    Args:
    :manifest: a JSON manifest as a list of items, each item describes 1 utt.
    :data_dir: a data directory path where to search for audio files, needed if
        the manifest contains relative paths
    """
    if data_dir is None:
        data_dir = Path("/")
    else:
        data_dir = Path(data_dir)

    files = []
    for item in manifest:
        fpath = data_dir / item["files"][0]["fname"]
        file_info = sox.file_info.info(str(fpath))

        # Check audio file exists
        assert Path(fpath).is_file(), f"file {fpath} does not exist"

        # Check transcript is not empty
        assert item["transcript"], f"{fpath} transcript is empty"

        # Check relevant audio metadata
        assert (
            item["original_duration"] == file_info["duration"]
        ), f"{fpath} faulty duration"
        assert (
            item["original_num_samples"] == file_info["num_samples"]
        ), f"{fpath} faulty number of samples"

        files.append(fpath)

    # Check there are no duplicate items
    assert len(set(files)) == len(files), "duplicate items in manifest"


def process_utterance(data: Dict, data_dir: Union[str, Path, None] = None) -> Dict:
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
    audio_file, trans = data.values()
    audio_file = data_dir / Path(audio_file)

    file_info = sox.file_info.info(str(audio_file))
    file_info["fname"] = str(audio_file)

    return {
        "transcript": trans,
        "files": [file_info],
        "original_duration": file_info["duration"],
        "original_num_samples": file_info["num_samples"],
    }


def prepare_manifest(
    data: List[Dict], num_jobs: int = 1, data_dir: Union[str, Path, None] = None
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
    with multiprocessing.Pool(num_jobs) as pool:
        dataset = pool.starmap(process_utterance, [(item, data_dir) for item in data])
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
