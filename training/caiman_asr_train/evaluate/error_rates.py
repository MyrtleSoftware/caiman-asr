#!/usr/bin/env python3

from enum import Enum

from beartype import beartype

from caiman_asr_train.data.text.has_spaces import (
    split_into_logical_tokens,
    warn_about_characters,
)


class ErrorRate(Enum):
    WORD = 1
    CHAR = 2
    MIXTURE = 3


@beartype
def get_error_rate(cfg: dict) -> ErrorRate:
    match cfg["error_rate"].lower():
        case "wer" | "word":
            return ErrorRate.WORD
        case "cer" | "char":
            return ErrorRate.CHAR
        case "mer" | "mixture":
            return ErrorRate.MIXTURE
        case _:
            raise ValueError(f"Invalid error rate: {cfg['error_rate']}")


@beartype
def error_rate_abbrev(error_rate: ErrorRate) -> str:
    match error_rate:
        case ErrorRate.WORD:
            return "wer"
        case ErrorRate.CHAR:
            return "cer"
        case ErrorRate.MIXTURE:
            return "mer"
        case _:
            raise ValueError(f"Invalid error rate: {error_rate}")


@beartype
def error_rate_long(error_rate: ErrorRate) -> str:
    match error_rate:
        case ErrorRate.WORD:
            return "Word Error Rate"
        case ErrorRate.CHAR:
            return "Character Error Rate"
        case ErrorRate.MIXTURE:
            return "Mixture Error Rate"
        case _:
            raise ValueError(f"Invalid error rate: {error_rate}")


@beartype
def decide_and_split(text: str, error_rate: ErrorRate) -> list[str]:
    warn_about_characters(text)
    match error_rate:
        case ErrorRate.WORD:
            return text.split()
        case ErrorRate.CHAR:
            return " ".join(text).split()
        case ErrorRate.MIXTURE:
            return split_into_logical_tokens(text)
        case _:
            raise ValueError(f"Invalid error rate: {error_rate}")
