#!/usr/bin/env python3

from dataclasses import dataclass

from beartype import beartype
from beartype.typing import List, Literal

from caiman_asr_train.utils.user_tokens_lite import get_all_user_tokens


@beartype
@dataclass
class Replacement:
    old: str
    new: str


@beartype
@dataclass
class NormalizeConfig:
    replacements: list[Replacement]
    user_symbols: list[str]
    user_symbols_dict: dict[str, str]
    standardize_text: bool
    preserve_case: bool
    preserve_punctuation: bool


@beartype
def normalize_config_from_full_yaml(
    model_config: dict, mode: Literal["train", "val"] = "train"
) -> NormalizeConfig:
    config_data = model_config[f"input_{mode}"]["audio_dataset"]
    user_symbols_dict, user_symbols = get_all_user_tokens(model_config)
    return normalize_config_from_config_data(
        config_data, user_symbols, user_symbols_dict, mode=mode
    )


@beartype
def normalize_config_from_config_data(
    config_data: dict,
    user_symbols: List[str],
    user_symbols_dict: dict[str, str],
    mode: Literal["train", "val"] = "train",
) -> NormalizeConfig:
    return get_normalize_config(
        config_data.get("replacements"),
        user_symbols,
        user_symbols_dict,
        # Whisper-style standardization (incl. filler-word stripping) defaults
        # to on for training and off for validation: validation's on-the-fly
        # normalization feeds the min_tokens cutset filter, so stripping a
        # filler-only reference to '' would silently drop that utterance from
        # the validation set instead of just scoring it against ''.
        config_data.get("standardize_text", mode == "train"),
        config_data.get("preserve_case", True),
        config_data.get("preserve_punctuation", True),
    )


@beartype
def get_normalize_config(
    replacements_config: list[dict[str, str]] | None,
    user_symbols: list[str],
    user_symbols_dict: dict[str, str],
    standardize_text: bool,
    preserve_case: bool,
    preserve_punctuation: bool,
) -> NormalizeConfig:
    replacements = (
        []
        if replacements_config is None
        else [Replacement(**r) for r in replacements_config]
    )

    return NormalizeConfig(
        replacements=replacements,
        user_symbols=user_symbols,
        user_symbols_dict=user_symbols_dict,
        standardize_text=standardize_text,
        preserve_case=preserve_case,
        preserve_punctuation=preserve_punctuation,
    )
