#!/usr/bin/env python3
from dataclasses import dataclass

from beartype import beartype
from beartype.typing import List, Optional

from caiman_asr_train.setup.text_normalization import (
    NormalizeConfig,
    normalize_config_from_config_data,
)
from caiman_asr_train.utils.select_transcripts import SelectTranscript


@beartype
@dataclass
class DaliYAMLConfig:
    select_transcripts: SelectTranscript
    max_duration: int | float
    min_duration: int | float
    sample_rate: int
    silence_threshold: Optional[int]
    resample_range: Optional[list[float]]
    prob_speed_perturbation: float
    window_size: float
    window_stride: float
    nfeatures: int
    nfft: int
    dither_coeff: float
    preemph_coeff: float
    normalize_config: NormalizeConfig
    max_transcript_len: int | float
    stats_path: str | None
    append_n_EOI: Optional[int]


@beartype
def build_dali_yaml_config(
    config_data: dict,
    config_features: dict,
    user_symbols: List[str],
    user_symbols_dict: dict[str, str],
    select_transcripts: SelectTranscript,
    total_time_stack_factor: int = -1,
) -> DaliYAMLConfig:
    if config_data["speed_perturbation"] is not None:
        resample_range = [
            config_data["speed_perturbation"]["min_rate"],
            config_data["speed_perturbation"]["max_rate"],
        ]
        prob_speed_perturbation = config_data["speed_perturbation"]["p"]
    else:
        resample_range = None
        prob_speed_perturbation = 0.0

    if config_data["append_EOI"]:
        assert total_time_stack_factor > 0

    return DaliYAMLConfig(
        select_transcripts=select_transcripts,
        max_duration=config_data["max_duration"],
        min_duration=config_data["min_duration"],
        sample_rate=config_data["sample_rate"],
        silence_threshold=-60 if config_data["trim_silence"] else None,
        resample_range=resample_range,
        prob_speed_perturbation=prob_speed_perturbation,
        window_size=config_features["window_size"],
        window_stride=config_features["window_stride"],
        nfeatures=config_features["n_filt"],
        nfft=config_features["n_fft"],
        dither_coeff=config_features["dither"],
        preemph_coeff=0.97,
        normalize_config=normalize_config_from_config_data(
            config_data, user_symbols, user_symbols_dict
        ),
        max_transcript_len=config_data["max_transcript_len"],
        stats_path=config_features.get("stats_path"),
        append_n_EOI=total_time_stack_factor if config_data["append_EOI"] else None,
    )
