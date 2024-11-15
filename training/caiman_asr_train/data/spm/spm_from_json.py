#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import sentencepiece as spm
import yaml

from caiman_asr_train.data.make_datasets.librispeech import LIBRISPEECH_TRAIN960H
from caiman_asr_train.data.text.preprocess import norm_and_tokenize_parallel
from caiman_asr_train.setup.text_normalization import normalize_config_from_full_yaml
from caiman_asr_train.utils.fast_json import fast_read_json


def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--spm_size", type=int, required=True)
    parser.add_argument("--spm_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument(
        "--train_manifests",
        type=str,
        nargs="+",
        default=LIBRISPEECH_TRAIN960H,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Where to save the spm, defaults to current directory.",
    )
    return parser


def get_transcripts(data_dir, manifest):
    name = Path(data_dir) / manifest
    data = fast_read_json(name)
    return [x["transcript"] for x in data]


def main(args):
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)

    normalize_config = normalize_config_from_full_yaml(model_config)

    allowed_chars = model_config["tokenizer"]["labels"]
    print(f"Allowed characters: {list(allowed_chars)}")
    transcripts = [
        transcript
        for manifest in args.train_manifests
        for transcript in get_transcripts(args.data_dir, manifest)
    ]

    normalized_transcripts = norm_and_tokenize_parallel(
        transcripts=transcripts,
        normalize_config=normalize_config,
        tokenizer=None,
        charset=allowed_chars,
    )

    tmp_file = Path("/tmp/normalized_transcripts.txt")

    with open(tmp_file, "w") as f:
        for x in normalized_transcripts:
            f.write(x + "\n")

    spm.SentencePieceTrainer.train(
        input=tmp_file,
        model_prefix=args.spm_name,
        vocab_size=args.spm_size,
        character_coverage=1.0,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        model_type="unigram",
        train_extremely_large_corpus=True,
        user_defined_symbols=",".join(normalize_config.user_symbols),
    )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for suffix in [".model", ".vocab"]:
            spm_file = Path(f"{args.spm_name}{suffix}")
            shutil.move(spm_file, output_dir)


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    main(args)
