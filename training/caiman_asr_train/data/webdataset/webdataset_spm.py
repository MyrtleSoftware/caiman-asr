import argparse
from argparse import Namespace
from tempfile import NamedTemporaryFile

import sentencepiece as spm

from caiman_asr_train.data.webdataset import WebDatasetReader
from caiman_asr_train.rnnt import config
from caiman_asr_train.rnnt.config import get_tokenizer_conf
from caiman_asr_train.setup.text_normalization import normalize_config_from_full_yaml


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", required=True, type=str, help="Root dir of dataset"
    )
    parser.add_argument(
        "--train_tar_files",
        required=True,
        type=str,
        nargs="+",
        help="One or more paths or globs for the training dataset tar files.",
    )
    parser.add_argument("--spm_name", type=str, required=True)
    parser.add_argument("--spm_size", type=int, default=8703)
    parser.add_argument(
        "--model_config",
        default="configs/base-8703sp_run.yaml",
        type=str,
        required=True,
        help="Path of the model configuration file",
    )
    return parser.parse_args()


def create_webdataset_spm(args):
    cfg = config.load(args.model_config)

    normalize_config = normalize_config_from_full_yaml(cfg)

    wds = WebDatasetReader(
        tokenizer=None,
        charset=get_tokenizer_conf(cfg)["labels"],
        shuffle=False,
        file_root=args.dataset_dir,
        tar_files=args.train_tar_files,
        normalize_config=normalize_config,
        skip_audio=True,
    )
    with NamedTemporaryFile("w", suffix=".txt") as f:
        for _, transcript, _, _ in wds:
            f.write(transcript + "\n")

        f.flush()
        spm.SentencePieceTrainer.train(
            input=f.name,
            model_prefix=args.spm_name,
            vocab_size=args.spm_size,
            character_coverage=1.0,
            bos_id=-1,
            eos_id=-1,
            model_type="unigram",
            train_extremely_large_corpus=True,
            user_defined_symbols=",".join(normalize_config.user_symbols),
        )


if __name__ == "__main__":
    args = parse_args()

    create_webdataset_spm(args)
