#!/usr/bin/env python3
import uuid
from argparse import ArgumentParser, Namespace
from functools import partial
from itertools import starmap
from pathlib import Path
from time import strftime

import datasets
import filetype
from beartype import beartype
from more_itertools import chunked
from tqdm import tqdm

from caiman_asr_train.data.hugging_face.core import check_transcript_key
from caiman_asr_train.data.make_datasets.manifest import prepare_manifest, save_manifest
from caiman_asr_train.data.make_datasets.sox_utils import convert_to_standard_format


@beartype
def make_argparser() -> ArgumentParser:
    parser = ArgumentParser("Converts a Hugging Face dataset to JSON format")
    parser.add_argument(
        "--hugging_face_dataset",
        "--hf_dataset",
        type=str,
        help="The name of the Hugging Face dataset",
        required=True,
    )
    parser.add_argument(
        "--hugging_face_config",
        "--hf_config",
        type=str,
        default=None,
        help="""When loading the dataset from the Hugging Face Hub,
        this option allows you to specify the configuration if needed.
        Defaults to None""",
    )
    parser.add_argument(
        "--hugging_face_split",
        "--hf_split",
        type=str,
        default="train",
        help="""The split to use. Defaults to 'train'.
        This supports the TensorFlow Slicing API, e.g. 'train[25%%:75%%]'
        will use the middle two quarters of the training set.
        See https://www.tensorflow.org/datasets/splits for more info
        """,
    )
    parser.add_argument(
        "--hugging_face_transcript_key",
        "--hf_transcript_key",
        type=str,
        default="text",
        help="""The key in the Hugging Face dataset that refers to the transcript.
        Defaults to "text", but could be something else, like "transcription".
        To find out the key, please look at the dataset's documentation
        on the Hugging Face Hub""",
    )
    parser.add_argument(
        "--data_dir",
        "--dataset_dir",
        type=str,
        help="The dataset will be saved as a child of this directory",
        required=True,
    )
    parser.add_argument(
        "--max_utterances_per_json",
        type=int,
        default=100000,
        help="For datasets with more than this many utterances, the manifests "
        "will be split across several JSON files. Default=100000",
    )
    parser.add_argument(
        "--max_leaf_dir_audios",
        type=int,
        default=100,
        help="Audios will be organized as data_dir/dataset_name/branch/leaf/audio.flac. "
        "This is is the maximum number of audios in leaf. Default=100",
    )
    parser.add_argument(
        "--max_branch_dir_audios",
        type=int,
        default=100,
        help="Audios will be organized as data_dir/dataset_name/branch/leaf/audio.flac. "
        "This is is the maximum number of leaves in branch. Default=100",
    )
    parser.add_argument(
        "--num_jobs_manifest_preparation",
        default=8,
        type=int,
        help="Number of parallel jobs for manifest preparation. Default=8",
    )
    parser.add_argument(
        "--fallback_input_audio_extension",
        type=str,
        default=None,
        help="Usually the code can detect the input audio format. "
        "If you think that will fail, you can specify a backup extension here.",
    )

    return parser


def timeprint(string):
    print(strftime("%c"), string)


@beartype
def get_extension(bytes_: bytes, fallback: str | None) -> str:
    ext = fallback if (info := filetype.guess(bytes_)) is None else info.extension
    assert (
        ext is not None
    ), "Could not determine input audio file format. Try --fallback_input_audio_extension"
    return ext


@beartype
def convert_utterance(
    args: Namespace,
    combined_name: str,
    i: int,
    hugging_face_elt: dict,
) -> dict[str, str]:
    transcript = hugging_face_elt[args.hugging_face_transcript_key]

    branch_dir = i // (args.max_leaf_dir_audios * args.max_branch_dir_audios)
    leaf_dir = (i // args.max_leaf_dir_audios) % args.max_branch_dir_audios

    audio_parent = Path(args.data_dir) / combined_name / str(branch_dir) / str(leaf_dir)
    audio_parent.mkdir(parents=True, exist_ok=True)
    audio_path = str(audio_parent / f"{i}.flac")

    if (bytes_ := hugging_face_elt["audio"]["bytes"]) is not None:
        ext = get_extension(bytes_, args.fallback_input_audio_extension)
        intermediate = Path("/tmp") / f"{uuid.uuid4()}.{ext}"
        with open(intermediate, "wb") as f:
            f.write(bytes_)
        convert_to_standard_format(str(intermediate), audio_path)
        intermediate.unlink()
    else:
        # If Hugging Face doesn't provide the raw bytes,
        # then it provides the absolute path to the audio file
        # https://github.com/huggingface/datasets/blob/main/src/datasets/features/audio.py#L24
        input_path = hugging_face_elt["audio"]["path"]
        convert_to_standard_format(input_path, audio_path)

    return {"audio_file": audio_path, "transcript": transcript}


@beartype
def make_json(
    path_transcript_list: list[dict],
    json_idx: int,
    combined_name: str,
    num_jsons: int,
    args: Namespace,
) -> None:
    timeprint("Creating json...")
    manifest = prepare_manifest(
        path_transcript_list,
        args.num_jobs_manifest_preparation,
        data_dir=args.data_dir,
    )
    timeprint("Writing json...")
    manifest_path = (
        Path(args.data_dir) / f"{combined_name}_{json_idx+1}_of_{num_jsons}.json"
    )
    save_manifest(manifest, manifest_path)


def main(args):
    timeprint("Loading Hugging face dataset...")
    dataset = datasets.load_dataset(
        args.hugging_face_dataset,
        name=args.hugging_face_config,
        split=args.hugging_face_split,
    ).cast_column("audio", datasets.Audio(decode=False))

    check_transcript_key(
        dataset, args.hugging_face_transcript_key, args.hugging_face_dataset
    )

    combined_name = (
        f"{args.hugging_face_dataset}_{args.hugging_face_config}_"
        f"{args.hugging_face_split}_{args.hugging_face_transcript_key}"
    ).replace("/", "_")

    timeprint("Writing audio files...")
    convert_utt = partial(convert_utterance, args, combined_name)

    flac_dataset = starmap(convert_utt, enumerate(tqdm(dataset)))
    batched_dataset = chunked(flac_dataset, args.max_utterances_per_json)
    num_jsons = (len(dataset) - 1) // args.max_utterances_per_json + 1
    for json_idx, path_transcript_list in enumerate(batched_dataset):
        make_json(path_transcript_list, json_idx, combined_name, num_jsons, args)

    timeprint("Done")


if __name__ == "__main__":
    parser = make_argparser()
    args = parser.parse_args()
    main(args)
