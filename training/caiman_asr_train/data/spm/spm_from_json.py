#!/usr/bin/env python3
import argparse
import os
import shutil
from argparse import Namespace
from pathlib import Path

import sentencepiece as spm
import torch.distributed as dist
import yaml
from beartype import beartype
from tqdm import tqdm

from caiman_asr_train.lhotse.config_build import build_from_legacy_sources
from caiman_asr_train.lhotse.sampler import get_lhotse_sampler_from_config
from caiman_asr_train.lhotse.stats import FiniteStats, InfiniteStats
from caiman_asr_train.setup.text_normalization import normalize_config_from_full_yaml
from caiman_asr_train.setup.tokenizer_labels import set_tokenizer_labels
from caiman_asr_train.train_utils.distributed import (
    get_rank_or_zero,
    get_world_size_or_one,
    merge_transcripts_across_ranks,
    start_ddp_with_gpu,
)
from caiman_asr_train.train_utils.torchrun import maybe_restart_with_torchrun


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spm_size", type=int, required=True)
    parser.add_argument("--spm_name", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument(
        "--train_dataset_yaml",
        type=str,
        required=True,
        help="Path to training dataset config (.yaml) file",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Where to save the spm, defaults to current directory.",
    )
    parser.add_argument(
        "--input_sentence_size",
        type=int,
        help="Number of sentences to use for SPM training",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        required=True,
        help="Number of GPUs to use for dataloading",
    )
    parser.add_argument(
        "--local_rank",
        default=os.getenv("LOCAL_RANK", 0),
        type=int,
        help="GPU id used for distributed processing",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Run data-prep on CPU",
    )
    parser.add_argument(
        "--called_by_torchrun",
        action="store_true",
        help="When a multiGPU script is split into multiple processes, "
        "this flag is set so that the processes don't try to split again.",
    )
    return parser.parse_args()


@beartype
def main(args: Namespace):
    # Fill in missing args
    args.seed = 42
    args.conditional = False
    args.prob_background_noise = 0.0
    args.prob_room_aug = 0.0
    args.sampling_mode = "fixed"
    args.batch_size = 64
    args.loader_workers_per_gpu = None
    args.turn_off_initial_padding = False
    args.inspect_audio = False
    args.norm_over_utterance = True
    args.eoi_drop_fraction = 0.5

    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)

    # ft_langs (if present) restricts training data to a subset of `langs`
    # for fine-tuning a resumed model. The SPM tokenizer must instead always
    # be built from all of `langs`, so that it matches the original model.
    model_config.pop("ft_langs", None)

    model_config = set_tokenizer_labels(model_config)

    world_size = get_world_size_or_one()
    global_rank = get_rank_or_zero()

    conf, *_ = build_from_legacy_sources(
        args,
        model_config,
        "train",
        use_cuda=False,  # Text-only processing, no need for CUDA
        global_rank=global_rank,
        world_size=world_size,
    )

    assert conf.sampler_shuffle is not None

    sampler, stats = get_lhotse_sampler_from_config(
        conf, global_rank=global_rank, world_size=world_size, tokenizer=None
    )

    match stats:
        case FiniteStats(total_utt=total_utt):
            est_utt = total_utt
        case InfiniteStats(max_orbit_utt=max_orbit_utt):
            est_utt = max_orbit_utt

    if args.input_sentence_size is None:
        args.input_sentence_size = float("inf")

    est_utt = int(
        min(est_utt, args.input_sentence_size) / (world_size * args.batch_size)
    )

    tmp_file = Path(f"/tmp/normalized_transcripts.rank{global_rank}.txt")

    bounded = zip(range(est_utt), sampler)

    with open(tmp_file, "w") as f:
        f.writelines(
            f"{seg.normalized_text}\n"
            for _, cuts in tqdm(
                bounded,
                desc="Getting transcripts",
                total=est_utt,
                disable=(global_rank != 0),
            )
            for cut in cuts.data
            for seg in cut.supervisions
        )

    if dist.is_initialized():
        dist.barrier()

    if global_rank == 0:
        merge_transcripts_across_ranks(
            [f"/tmp/normalized_transcripts.rank{i}.txt" for i in range(world_size)],
            output_path="/tmp/normalized_transcripts.txt",
        )

    if global_rank == 0:
        required_chars = list(set(model_config["tokenizer"]["labels"]) - set(" "))
        spm.SentencePieceTrainer.train(
            input="/tmp/normalized_transcripts.txt",
            model_prefix=args.spm_name,
            vocab_size=args.spm_size,
            character_coverage=1.0,
            required_chars=required_chars,
            unk_id=0,
            bos_id=-1,
            eos_id=-1,
            model_type="unigram",
            train_extremely_large_corpus=True,
            user_defined_symbols=",".join(
                normalize_config_from_full_yaml(model_config, mode="train").user_symbols
            ),
        )

        if args.output_dir is not None:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for suffix in [".model", ".vocab"]:
                spm_file = Path(f"{args.spm_name}{suffix}")
                shutil.move(spm_file, output_dir)


if __name__ == "__main__":
    args = parse_args()
    if not args.cpu:
        maybe_restart_with_torchrun(
            args.num_gpus,
            args.called_by_torchrun,
            "/workspace/training/caiman_asr_train/data/spm/spm_from_json.py",
        )
    if not args.cpu:
        multi_gpu = args.num_gpus > 1
        _ = start_ddp_with_gpu(
            multi_gpu,
            args.local_rank,
            skip_init=False,
            timeout={"weeks": 52},
        )
    main(args)
