from argparse import Namespace

from caiman_asr_train.data.dali.mel_normalization import MelFeatNormalizer, NormType
from caiman_asr_train.data.mel_stats import MelStats
from caiman_asr_train.setup.core import PipelineType
from caiman_asr_train.setup.dali import DaliYAMLConfig


def build_mel_feat_normalizer(
    args: Namespace,
    dali_yaml_config: DaliYAMLConfig,
    pipeline_type: PipelineType,
    batch_size: int,
) -> MelFeatNormalizer:
    """
    Build mel feature normalizer.
    """
    if pipeline_type == PipelineType.TRAIN:
        norm_type = NormType.BLENDED_STATS
    elif pipeline_type == PipelineType.VAL:
        norm_type = NormType.DATASET_STATS
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

    if args.norm_over_utterance:
        # then override the norm_type
        norm_type = NormType.UTTERANCE_STATS

    if args.norm_use_global_stats:
        # then override the norm_type
        norm_type = NormType.DATASET_STATS

    mel_stats = (
        MelStats.from_dir(dali_yaml_config.stats_path)
        if norm_type != NormType.UTTERANCE_STATS
        else None
    )

    ramp_start, ramp_end = norm_ramp_params(norm_type, args)
    return MelFeatNormalizer(
        mel_stats=mel_stats,
        ramp_start_step=ramp_start,
        ramp_end_step=ramp_end,
        batch_size=batch_size,
        starting_ratio=args.norm_starting_ratio,
        norm_type=norm_type,
    )


def norm_ramp_params(norm_type: NormType, args: Namespace, default_ramp: int = 5000):
    """
    Return Tuple of start, end steps for the BLENDED_STATS ramp period.

    The default schedule uses a heuristic that says the ramp should start after the
    learning rate has decayed to half of its original value and continue for 5k steps.
    In Myrtle.ai experiments this heuristic gave stable training and low WERs.
    """
    if norm_type != NormType.BLENDED_STATS:
        return None, None

    if args.norm_ramp_start_step is not None:
        start_step = args.norm_ramp_start_step
    else:
        start_step = args.warmup_steps + args.hold_steps + args.half_life_steps
    if args.norm_ramp_end_step is not None:
        end_step = args.norm_ramp_end_step
    else:
        end_step = start_step + default_ramp
    return start_step, end_step
