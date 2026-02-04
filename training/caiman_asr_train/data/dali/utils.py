#! /usr/bin/env python3

import json
import random
import tempfile
from collections import defaultdict
from operator import itemgetter
from pathlib import Path

import sox
from beartype import beartype
from beartype.typing import Callable, Dict, Optional, Tuple, Union

from caiman_asr_train.train_utils.distributed import print_once
from caiman_asr_train.utils.fast_json import fast_read_json
from caiman_asr_train.utils.select_transcripts import (
    _DEFAULT_TRANSCRIPT,
    SelectTranscript,
)


@beartype
def generate_json(
    audio_list: list[Path],
    txt_list: list[Path],
    datadir_path: Path,
) -> str:
    """
    Generate json file with MLPerf Schema

    Example of the generated json file:
    [
          {
            "transcript": "brief i pray for you for you see tis a busy time with",
            "files": [
              {
                "fname": "train-clean-100/1502/122619/1502-122619-0000.flac",
              }
            ],
            "original_duration": 10.74,
          },
     ]

    Args:
        audio_list: list of audio files
        txt_list: list of txt files
        datadir_path: path to the dataset directory

    Returns:
        output_file_name: path to the generated json file

    """
    all_transcripts = []
    for audio, txt in zip(audio_list, txt_list):
        dictionary = {}
        with open(txt, "r") as f:
            text = f.readline().strip("\n")
            dictionary[_DEFAULT_TRANSCRIPT] = text
        dictionary["files"] = [{"fname": str(audio.relative_to(datadir_path))}]
        file_info = sox.file_info.info(str(audio))
        dictionary["original_duration"] = file_info["duration"]
        all_transcripts.append(dictionary)
    # generate json temporary file
    json_tempfile = tempfile.NamedTemporaryFile(mode="w", delete=False)
    with open(json_tempfile.name, "w") as f:
        json.dump(all_transcripts, f, indent=4)
    return json_tempfile.name


@beartype
def generate_json_names_from_dirs(
    dataset_path: str, audio_dir: str, txt_dir: str
) -> list[str]:
    # get directories absolute paths
    datadir_path = Path(dataset_path)
    audio_path = datadir_path.joinpath(audio_dir)
    txt_path = datadir_path.joinpath(txt_dir)

    # generate list of audio and txt files
    audio_list = get_path_files(audio_path, [".wav", ".flac"], strict=True)
    txt_list = get_path_files(txt_path, [".txt"], strict=True)

    # generate temporary json file
    json_file_path = generate_json(audio_list, txt_list, datadir_path)

    return [json_file_path]


@beartype
def get_path_files(
    dir_path: Path, valid_suffixes: list = None, strict: bool = True
) -> list[Path]:
    full_list = [f for f in dir_path.glob("**/*") if f.is_file()]
    full_list.sort()
    if strict:
        full_list = [i for i in full_list if i.suffix in valid_suffixes]
    return full_list


@beartype
def set_predicate(
    max_duration: int | float,
    max_transcript_len: int | float,
    min_duration: int | float = 0.05,
) -> Callable[[dict, str], bool]:
    """
    Returns a function that decides whether an utterance is short enough
    for the dataset. Typically in validation this will always return true
    since max_duration == max_transcript_len == float("inf")

    Args:
        max_duration: maximum duration of the audio file
        max_transcript_len: maximum length of the transcript
        min_duration: minimum duration of the audio file
    """

    return (
        lambda json, transcript: json["original_duration"] <= max_duration
        and json["original_duration"] > min_duration
        and (len(transcript) < max_transcript_len)
    )


@beartype
def _parse_json(
    json_path: str,
    start_label: int = 0,
    predicate=lambda json, transcript: True,
    use_transcript: SelectTranscript = SelectTranscript(),
) -> Tuple[Dict[str, Dict[str, Union[int, float]]], Dict[int, str]]:
    """
    Parses json file to the format required by DALI
    Args:
        json_path: path to json file
        start_label: the label, starting from which DALI will assign consecutive int
            numbers to every transcript
        predicate: function, that accepts a sample descriptor (i.e. json dictionary)
            and transcript as an argument. If the predicate for a given sample
            returns True, it will be included in the dataset.
        use_transcript: The transcript selector

    Returns:
        output_files: dictionary, that maps file name to the label and duration
            assigned by DALI
        transcripts: dictionary, that maps label assigned by DALI to the transcript
    """
    manifest = fast_read_json(json_path)
    output_files = {}
    transcripts = {}
    curr_label = start_label

    reasons = defaultdict(int)

    for original_sample in manifest:
        # Select has error handling
        transcript = use_transcript(original_sample)

        match use_transcript.select(original_sample):
            case None:
                # If mode is skip and transcript is not found, skip the sample
                reasons["missing-transcript-key"] += 1
            case (transcript, key) if not predicate(original_sample, transcript):
                reasons[f"bad-length@{key}"] += 1
            case (transcript, key):
                reasons[key] += 1

                transcripts[curr_label] = transcript
                fname = original_sample["files"][-1]["fname"]
                output_files[fname] = dict(
                    label=curr_label,
                    duration=original_sample["original_duration"],
                )
                curr_label += 1

    print_once(f"In {json_path}:")
    for key, count in sorted(reasons.items(), key=itemgetter(1), reverse=True):
        print_once(f"\t{count:>12} ({count / len(manifest):6.2%}) {key}")

    return output_files, transcripts


@beartype
def _filter_files(
    output_files: Dict[str, Dict[str, Union[int, float]]],
    transcripts: Dict[int, str],
    n_utterances_only: Optional[int],
    seed: int,
) -> Tuple[Dict[str, Dict[str, Union[int, float]]], Dict[int, str]]:
    """Shuffles the dataset and takes only n_utterances_only utterances"""
    # Inside this function use the same seed across all processes,
    # since the dataset sharding depends on output_files and transcripts
    # being identical across processes.
    generator = random.Random(seed)
    if n_utterances_only is None:
        return output_files, transcripts
    how_many_to_take = min(n_utterances_only, len(output_files))
    output_files_sublist = generator.sample(
        list(output_files.items()), how_many_to_take
    )
    transcripts_sublist = [
        (utt_info["label"], transcripts[utt_info["label"]])
        for _, utt_info in output_files_sublist
    ]
    # The label idxs have to be a permutation of 0...length-1, so overwrite
    # the existing label idxs
    transcripts_good_indices = [
        (i, transcript) for i, (_, transcript) in enumerate(transcripts_sublist)
    ]
    output_files_good_indices = [
        (fname, {"label": i, "duration": utt_info["duration"]})
        for i, (fname, utt_info) in enumerate(output_files_sublist)
    ]
    return dict(output_files_good_indices), dict(transcripts_good_indices)
