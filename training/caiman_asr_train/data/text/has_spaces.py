#!/usr/bin/env python3
import unicodedata

from beartype import beartype


@beartype
def is_from_spaceless_language(char: str) -> bool:
    # TODO This is an imperfect heuristic and will
    # miss some languages that don't use spaces,
    # e.g. Thai and Javanese. See:
    # https://en.wikipedia.org/wiki/Scriptio_continua
    # https://en.wikipedia.org/wiki/Category:Writing_systems_without_word_boundaries
    return is_in_cjk_block(char)


@beartype
def pad_with_space(char: str) -> str:
    return f" {char} " if is_from_spaceless_language(char) else char


@beartype
def split_into_logical_tokens(strng: str) -> list[str]:
    return "".join(pad_with_space(char) for char in strng).split()


@beartype
def is_in_cjk_block(char: str) -> bool:
    return unicodedata.name(char, "").startswith("CJK ")


@beartype
def is_ascii_or_cjk(char: str) -> bool:
    return ord(char) < 128 or is_in_cjk_block(char)


@beartype
def warn_about_characters(text: str) -> None:
    if not all(is_ascii_or_cjk(char) for char in text):
        print(
            f"Warning: some characters in {text} are not ASCII or CJK. "
            "If you are training on a language other than English, "
            "please contact caiman-asr@myrtle.ai for support"
        )
