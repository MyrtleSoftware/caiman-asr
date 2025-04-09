#!/usr/bin/env python3

from beartype import beartype

from caiman_asr_train.data.build_dataloader import build_dali_loader
from caiman_asr_train.data.dali.data_loader import DaliDataLoader
from caiman_asr_train.rnnt import config
from caiman_asr_train.setup.core import PipelineType
from caiman_asr_train.setup.dali import build_dali_yaml_config
from caiman_asr_train.setup.mel_normalization import build_mel_feat_normalizer
from caiman_asr_train.utils.user_tokens_lite import get_all_user_tokens


@beartype
def build_dataloader_util(
    dataload_args,
    pipeline_type,
    batch_size,
    mini_config_fp,
    tokenizer,
    train_sampler=None,
    deterministic_ex_noise: bool = False,
    max_transcript_len: int = 450,
    normalize: bool = True,
    min_duration: float = 0.05,
) -> DaliDataLoader:
    """
    Build dali dataloader helper function for testing.
    """
    cfg = config.load(mini_config_fp)
    cfg["input_train"]["audio_dataset"]["max_transcript_len"] = max_transcript_len
    user_symbols = list(get_all_user_tokens(cfg).values())
    dataset_kw, features_kw, _, _ = config.input(cfg, pipeline_type)

    if deterministic_ex_noise:
        # make dataloader deterministic except for noise augmentation
        features_kw["dither"] = 0.0
        dataset_kw["speed_perturbation"] = None

    dali_yaml_config = build_dali_yaml_config(
        config_data=dataset_kw, config_features=features_kw, user_symbols=user_symbols
    )
    pipeline_type_enum = (
        PipelineType.TRAIN if pipeline_type == "train" else PipelineType.VAL
    )
    mel_feat_normalizer = (
        build_mel_feat_normalizer(
            dataload_args, dali_yaml_config, pipeline_type_enum, batch_size
        )
        if normalize
        else None
    )
    return build_dali_loader(
        dataload_args,
        pipeline_type,
        batch_size=batch_size,
        dali_yaml_config=dali_yaml_config,
        train_sampler=train_sampler,
        tokenizer=tokenizer,
        cpu=True,
        no_logging=True,
        mel_feat_normalizer=mel_feat_normalizer,
        world_size=1,
    )
