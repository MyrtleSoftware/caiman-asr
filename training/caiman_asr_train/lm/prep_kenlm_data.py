import argparse
import json
import os
import tarfile

import yaml
from beartype import beartype
from beartype.typing import List

from caiman_asr_train.data.make_datasets.librispeech import LIBRISPEECH_TRAIN960H
from caiman_asr_train.data.text.preprocess import norm_and_tokenize
from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.setup.text_normalization import (
    NormalizeConfig,
    normalize_config_from_full_yaml,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/datasets/LibriSpeech",
        help="Root dir of dataset",
    )
    parser.add_argument(
        "--manifests",
        type=str,
        default=LIBRISPEECH_TRAIN960H,
        nargs="+",
        help="""Paths of the training dataset manifest files.
        Ignored if --read_from_tar=True.""",
    )
    parser.add_argument(
        "--read_from_tar",
        action="store_true",
        default=False,
        help="Read data from tar files instead of json manifest files",
    )
    parser.add_argument(
        "--tar_files",
        type=str,
        required=False,
        default=None,
        nargs="+",
        help="Paths of the dataset tar files. Ignored if --read_from_tar=False.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output .txt file containing tokenized transcripts to train KenLM.",
        required=True,
    )
    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to model configuration yaml file.",
        required=True,
    )
    return parser.parse_args()


@beartype
def extract_transcripts_from_manifests(
    manifest_files: List[str],
    path: str,
    tokenizer: Tokenizer,
    normalize_config: NormalizeConfig,
    labels: List[str],
) -> List[List[int]]:
    """Read JSON manifests, and return normalized and tokenized transcripts."""
    transcripts = []
    num_files = len(manifest_files)
    for i, json_file in enumerate(manifest_files):
        fpath = os.path.join(path, json_file)
        print(f"Processing manifest {i+1}/{num_files}: {fpath}")
        with open(fpath, "r") as file:
            data = json.load(file)
        transcripts.extend(
            [
                norm_and_tokenize(
                    item["transcript"], normalize_config, tokenizer, charset=labels
                )
                for item in data
            ]
        )
    return transcripts


@beartype
def extract_transcripts_from_tars(
    tar_files: List[str],
    path: str,
    tokenizer: Tokenizer,
    normalize_config: NormalizeConfig,
    labels: List[str],
) -> List[List[int]]:
    transcripts = []
    num_files = len(tar_files)
    for i, tar_file in enumerate(tar_files):
        fpath = os.path.join(path, tar_file)
        print(f"Processing tar file {i+1}/{num_files}: {fpath}")
        with tarfile.open(fpath, "r") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".txt"):
                    txt_file = tar.extractfile(member)
                    txt = txt_file.read().decode()
                    transcripts.append(
                        norm_and_tokenize(txt, normalize_config, tokenizer, labels)
                    )
    return transcripts


@beartype
def save_transcripts_to_file(transcripts: List[str], filename: str):
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
    with open(filename, "w") as file:
        for transcript in transcripts:
            file.write(transcript + "\n")


def main():
    args = parse_args()
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)

    tokenizer_cfg = model_config["tokenizer"]
    labels = tokenizer_cfg["labels"]
    tokenizer = Tokenizer(
        labels, tokenizer_cfg["sentpiece_model"], tokenizer_cfg["sampling"]
    )
    normalize_config = normalize_config_from_full_yaml(model_config)

    if args.read_from_tar:
        transcripts = extract_transcripts_from_tars(
            args.tar_files, args.data_dir, tokenizer, normalize_config, labels
        )
    else:
        transcripts = extract_transcripts_from_manifests(
            args.manifests, args.data_dir, tokenizer, normalize_config, labels
        )
    # Convert tokenized transcripts to the expected form for KenLM,
    # where each sentence is a string of space-separated tokens.
    token_sentences = [
        " ".join(tokenizer.sentpiece.id_to_piece(token) for token in sentence)
        for sentence in transcripts
    ]
    save_transcripts_to_file(token_sentences, args.output_path)
    print(f"Saved tokenized transcripts to {args.output_path}")


if __name__ == "__main__":
    main()
