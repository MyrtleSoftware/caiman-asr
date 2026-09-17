import argparse
import os
from argparse import Namespace
from pathlib import Path

import torch.distributed as dist
import yaml
from tqdm import tqdm

from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.lhotse.config_build import build_from_legacy_sources
from caiman_asr_train.lhotse.sampler import get_lhotse_sampler_from_config
from caiman_asr_train.lhotse.stats import FiniteStats, InfiniteStats
from caiman_asr_train.rnnt.config import get_tokenizer_conf
from caiman_asr_train.setup.tokenizer_labels import set_tokenizer_labels
from caiman_asr_train.train_utils.distributed import (
    get_rank_or_zero,
    get_world_size_or_one,
    merge_transcripts_across_ranks,
    start_ddp_with_gpu,
)
from caiman_asr_train.train_utils.torchrun import maybe_restart_with_torchrun
from caiman_asr_train.utils.user_tokens_lite import get_all_user_tokens


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dataset_yaml",
        type=str,
        required=True,
        help="Path to training dataset config (.yaml) file",
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
    parser.add_argument(
        "--max_utterances",
        type=int,
        help="Maximum number of utterances to process.",
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
        help="""Run data-prep on CPU
        """,
    )
    parser.add_argument(
        "--called_by_torchrun",
        action="store_true",
        help="""When a multiGPU script is split into multiple processes, this flag is
        set so that the processes don't try to split again. The user should not set this
        flag""",
    )
    return parser.parse_args()


def main(args: Namespace):
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)

    # ft_langs (if present) restricts training data to a subset of `langs`
    # for fine-tuning a resumed model. The KenLM ngram model must instead
    # always be built from all of `langs`, so that it matches the original
    # model.
    model_config.pop("ft_langs", None)

    model_config = set_tokenizer_labels(model_config)

    # Currently user tokens are not passed to the LM during decoding.
    # Hence, they should be stripped from the model configuration file such
    # that they are normalized out of the transcripts.
    if tokens := get_all_user_tokens(model_config)[0]:
        print(f"INFO: stripping user {tokens=} from model configuration file.")
        del model_config["user_tokens"]
        assert not get_all_user_tokens(model_config)[0]

    tokenizer_cfg = get_tokenizer_conf(model_config)
    if tokenizer_cfg["label_smoothing_kwargs"] is not None:
        print("INFO: disabling label smoothing for KenLM.")
        tokenizer_cfg["label_smoothing_kwargs"] = None

    # Fill in missing args
    args.seed = args.seed if hasattr(args, "seed") else 42
    args.conditional = False
    args.prob_background_noise = 0.0
    args.prob_room_aug = 0.0
    args.sampling_mode = "fixed"
    args.batch_size = args.batch_size if hasattr(args, "batch_size") else 64
    args.loader_workers_per_gpu = None
    args.turn_off_initial_padding = False
    args.inspect_audio = False
    args.norm_over_utterance = True
    args.eoi_drop_fraction = 0.5

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

    if tokenizer_cfg["unk_handling"].value == "WARN":
        # When the user sets unk_handling to WARN, filter out transcripts
        # with unks so that KenLM training doesn't crash
        conf.filter_unk = 0

    tokenizer = Tokenizer(**tokenizer_cfg)

    sampler, stats = get_lhotse_sampler_from_config(
        conf, global_rank=global_rank, world_size=world_size, tokenizer=tokenizer
    )

    match stats:
        case FiniteStats(total_utt=total_utt):
            est_utt = total_utt
        case InfiniteStats(max_orbit_utt=max_orbit_utt):
            est_utt = max_orbit_utt

    if args.max_utterances is None:
        args.max_utterances = float("inf")

    # Limit number of utterances if specified
    est_utt = int(min(est_utt, args.max_utterances) / (world_size * args.batch_size))

    tmp_file = Path(f"/tmp/kenlm_transcripts.rank{global_rank}.txt")

    bounded = zip(range(est_utt), sampler)

    with open(tmp_file, "w") as f:
        f.writelines(
            f"{' '.join(tokenizer.sentpiece.id_to_piece(seg.tokens.tolist()))}\n"
            for _, cuts in tqdm(
                bounded,
                desc="Getting transcripts",
                total=est_utt,
                disable=(global_rank != 0),
            )
            for cut in cuts.data
            for seg in cut.supervisions
        )

    # Consolidate all files
    if dist.is_initialized():
        dist.barrier()

    if global_rank == 0:
        merge_transcripts_across_ranks(
            [f"/tmp/kenlm_transcripts.rank{i}.txt" for i in range(world_size)],
            output_path=args.output_path,
        )


if __name__ == "__main__":
    args = parse_args()
    if not args.cpu:
        maybe_restart_with_torchrun(
            args.num_gpus,
            args.called_by_torchrun,
            "/workspace/training/caiman_asr_train/lm/prep_kenlm_data.py",
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
