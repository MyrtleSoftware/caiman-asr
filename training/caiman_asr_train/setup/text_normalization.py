#!/usr/bin/env python3

from dataclasses import dataclass
from enum import Enum

from beartype import beartype
from beartype.typing import List

from caiman_asr_train.utils.user_tokens_lite import get_all_user_tokens


class NormalizeLevel(Enum):
    """See docs in changing_the_character_set.md"""

    IDENTITY = 1
    SCRUB = 2
    ASCII = 3
    DIGIT_TO_WORD = 4
    LOWERCASE = 5


@beartype
def get_normalize_level(level: str | bool) -> NormalizeLevel:
    # For compatibility with yamls from versions `<=1.10.0`
    if level is True:
        return NormalizeLevel.LOWERCASE
    elif level is False:
        return NormalizeLevel.IDENTITY
    return NormalizeLevel[level.upper()]


@beartype
@dataclass
class Replacement:
    old: str
    new: str


@beartype
@dataclass
class NormalizeConfig:
    normalize_level: NormalizeLevel
    replacements: list[Replacement]
    remove_tags: bool
    user_symbols: list[str]
    user_symbols_dict: dict[str, str]
    standardize_text: bool
    preserve_case: bool
    preserve_punctuation: bool


IDENTITY_NORMALIZE_CONFIG = NormalizeConfig(
    normalize_level=NormalizeLevel.IDENTITY,
    replacements=[],
    remove_tags=False,
    user_symbols=[],
    user_symbols_dict={},
    standardize_text=False,
    preserve_case=False,
    preserve_punctuation=False,
)


@beartype
def normalize_config_from_full_yaml(model_config: dict) -> NormalizeConfig:
    config_data = model_config["input_train"]["audio_dataset"]
    user_symbols_dict, user_symbols = get_all_user_tokens(model_config)
    return normalize_config_from_config_data(
        config_data, user_symbols, user_symbols_dict
    )


@beartype
def normalize_config_from_config_data(
    config_data: dict, user_symbols: List[str], user_symbols_dict: dict[str, str]
) -> NormalizeConfig:
    return get_normalize_config(
        config_data["normalize_transcripts"],
        config_data.get("replacements"),
        config_data.get("remove_tags", True),
        user_symbols,
        user_symbols_dict,
        config_data.get("standardize_text"),
        config_data.get("preserve_case", False),
        config_data.get("preserve_punctuation", False),
    )


@beartype
def get_normalize_config(
    level: str | bool,
    replacements_config: list[dict[str, str]] | None,
    remove_tags: bool,
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
        normalize_level=get_normalize_level(level),
        replacements=replacements,
        remove_tags=remove_tags,
        user_symbols=user_symbols,
        user_symbols_dict=user_symbols_dict,
        standardize_text=standardize_text,
        preserve_case=preserve_case,
        preserve_punctuation=preserve_punctuation,
    )
