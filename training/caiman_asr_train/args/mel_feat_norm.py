from argparse import ArgumentParser, Namespace

from beartype import beartype


@beartype
def add_mel_feat_norm_args(parser: ArgumentParser) -> None:
    mel_stat_args = parser.add_argument_group("MelFeatNormalizer setup")

    mel_stat_args.add_argument(
        "--norm_starting_ratio",
        type=float,
        default=0.0,
        help="The initial dataset_to_utt_ratio. This is a float in [0, 1] where 0 means "
        "only utterance-specific stats are used and 1 means only the dataset stats are "
        "used. See MelFeatNormalizer for more detail.",
    )
    mel_stat_args.add_argument(
        "--norm_ramp_start_step",
        type=int,
        default=None,
        help="The step at which the amount of dataset stats normalization starts to ramp "
        "up. If not provided, it is set to the point at which the learning rate has "
        "decayed to half of its original value.",
    )
    mel_stat_args.add_argument(
        "--norm_ramp_end_step",
        type=int,
        default=None,
        help="The step at which the amount of dataset stats normalization finishes "
        "ramping up. If not provided, it is set to (--norm_ramp_start_step + 5000).",
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
    if args.norm_starting_ratio < 0 or args.norm_starting_ratio > 1:
        raise ValueError(
            f"norm_starting_ratio must be in [0, 1], but got {args.norm_starting_ratio}"
        )

    if args.norm_ramp_start_step is not None and args.norm_ramp_start_step < 0:
        raise ValueError(
            "norm_ramp_start_step must be non-negative, but got "
            f"{args.norm_ramp_start_step}"
        )

    if args.norm_ramp_end_step is not None and args.norm_ramp_end_step < 0:
        raise ValueError(
            f"norm_ramp_end_step must be non-negative, but got {args.norm_ramp_end_step}"
        )

    if args.norm_ramp_start_step is not None and args.norm_ramp_end_step is not None:
        if args.norm_ramp_end_step < args.norm_ramp_start_step:
            raise ValueError(
                "norm_ramp_end_step must be greater than or equal to norm_ramp_start_step "
                f"but got norm_ramp_start_step={args.norm_ramp_start_step} and "
                f"norm_ramp_end_step={args.norm_ramp_end_step}"
            )
