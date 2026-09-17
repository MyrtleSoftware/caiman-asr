from argparse import ArgumentParser, Namespace

from beartype import beartype

from caiman_asr_train.train_utils.distributed import warn_once


@beartype
def add_mel_feat_norm_args(parser: ArgumentParser) -> None:
    mel_stat_args = parser.add_argument_group("MelFeatNormalizer setup")

    mel_stat_args.add_argument(
        "--norm_use_global_stats",
        action="store_true",
        help="Use global stats for normalization. Overrides other flags",
    )
    mel_stat_args.add_argument(
        "--norm_ramp_start",
        type=int,
        help="The duration when the dataset stats normalization starts to ramp",
    )
    mel_stat_args.add_argument(
        "--norm_ramp_end",
        type=int,
        help="The duration when the dataset stats normalization finishes ramping",
    )
    mel_stat_args.add_argument(
        "--norm_over_utterance",
        action="store_true",
        help="Normalize mel features with stats computed over the time dimension of the "
        "full utterance rather than the pre-generated dataset stats found at the "
        "--model_config stats_path. Note that this is not streaming compatible so it is "
        "not recommended to train a model with this option. It is kept for backwards "
        "compatibility in order to evaluate models trained on <=v1.8.0. (Previously, "
        "streaming norm was used at inference time but this was removed in v1.9.0)",
    )


def check_mel_feat_norm_args(args: Namespace) -> None:
    assert not (
        args.norm_over_utterance and args.norm_use_global_stats
    ), "--norm_over_utterance is --exclusive with norm_use_global_stats"

    if (
        "fine_tune" in args
        and args.fine_tune
        and not (args.norm_over_utterance or args.norm_use_global_stats)
    ):
        warn_once(
            "WARNING: Fine-tuning a model without using global stats or "
            "utterance stats for normalization may result in poor performance "
            "if there is a step change between the training and fine-tuning data."
        )

    if args.norm_use_global_stats:
        # Check that the user has not set any other normalization options.
        assert args.norm_ramp_start is None
        assert args.norm_ramp_end is None

    if args.norm_ramp_start is not None and args.norm_ramp_start < 0:
        raise ValueError(
            f"norm_ramp_start must be non-negative, but got {args.norm_ramp_start}"
        )

    if args.norm_ramp_end is not None and args.norm_ramp_end < 0:
        raise ValueError(
            f"norm_ramp_end must be non-negative, but got {args.norm_ramp_end}"
        )

    if args.norm_ramp_start is not None and args.norm_ramp_end is not None:
        if args.norm_ramp_end < args.norm_ramp_start:
            raise ValueError(
                "norm_ramp_end must be greater than or equal to norm_ramp_start "
                f"but got norm_ramp_start={args.norm_ramp_start} and "
                f"norm_ramp_end={args.norm_ramp_end}"
            )
