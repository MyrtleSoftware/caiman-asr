#!/usr/bin/env python3
from dataclasses import dataclass

from beartype import beartype
from beartype.typing import Optional


@beartype
@dataclass
class DaliYAMLConfig:
    max_duration: float
    sample_rate: int
    silence_threshold: Optional[int]
    resample_range: Optional[list[float]]
    window_size: float
    window_stride: float
    nfeatures: int
    nfft: int
    dither_coeff: float
    preemph_coeff: float
    normalize_transcripts: bool
    max_transcript_len: int | float
    stats_path: str | None


@beartype
def build_dali_yaml_config(config_data: dict, config_features: dict) -> DaliYAMLConfig:
    if config_data["speed_perturbation"] is not None:
        resample_range = [
            config_data["speed_perturbation"]["min_rate"],
            config_data["speed_perturbation"]["max_rate"],
        ]
    else:
        resample_range = None
    return DaliYAMLConfig(
        max_duration=config_data["max_duration"],
        sample_rate=config_data["sample_rate"],
        silence_threshold=-60 if config_data["trim_silence"] else None,
        resample_range=resample_range,
        window_size=config_features["window_size"],
        window_stride=config_features["window_stride"],
        nfeatures=config_features["n_filt"],
        nfft=config_features["n_fft"],
        dither_coeff=config_features["dither"],
        preemph_coeff=0.97,
        normalize_transcripts=config_data["normalize_transcripts"],
        max_transcript_len=config_data["max_transcript_len"],
        stats_path=config_features.get("stats_path"),
    )
