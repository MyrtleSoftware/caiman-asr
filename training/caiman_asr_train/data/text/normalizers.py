#!/usr/bin/env python3
import warnings

import inflect
from beartype import beartype
from text_unidecode import unidecode

from caiman_asr_train.data.text.is_tag import actually_remove_tags
from caiman_asr_train.data.text.ito import _clean_text, punctuation_map
from caiman_asr_train.data.text.ito.numbers import normalize_numbers
from caiman_asr_train.data.text.whisper_text_normalizer import EnglishTextNormalizer
from caiman_asr_train.setup.text_normalization import NormalizeConfig, NormalizeLevel


@beartype
def punctuation_normalize(text: str) -> str:
    """Normalize punctuation.

    This function keeps all the punctuation marks
    intact apart from the following:

    `;` -> `,` semi-colon to comma
    `:` -> `.` colon to period
    `!` -> `.` exclamation mark to period
    `-` -> ` ` hyphen to space

    Parameters
    ----------
    text
        string containing text to normalize

    Returns
    -------
    text
        string containing normalized text
    """
    text = text.replace('"', "")
    text = text.replace(";", ",")
    text = text.replace(":", ".")
    text = text.replace("!", ".")
    text = text.replace("-", " ")

    return text


@beartype
def lowercase_normalize(
    s, charset: list[str], quiet: bool = False, scrub: bool = True
) -> str:
    """Normalizes string.

    Example:
    >>> import string
    >>> lowercase_normalize('call me at 8:00 pm!', charset=list(string.ascii_lowercase+' '))
    'call me at eight zero zero pm'
    """
    # When called from the training pipeline, scrub should be False,
    # since the scrubbing should happen after the replacements.
    # When called from elsewhere, scrub=True for backwards compatibility.
    punct_map = punctuation_map(charset) if scrub else None

    try:
        text = _clean_text(s, ["english_cleaners"], charset, punct_map).strip()
    except (ValueError, inflect.NumOutOfRangeError, IndexError) as err:
        if not quiet:
            print(f"Expected {err=}, {type(err)=}")
            print(f"WARNING: Normalizing this line failed: {s}")
            print("Omitting the line that couldn't be normalized")
            print()
        return ""
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        print(f"ERROR: Normalizing this line failed: {s}")
        raise
    else:
        return text


@beartype
def select_and_normalize(
    transcript: str, charset: list[str], normalize_config: NormalizeConfig
) -> str:
    def replace(text: str) -> str:
        for r in normalize_config.replacements:
            text = text.replace(r.old, r.new)
        return text

    def remove_tags(text: str) -> str:
        return actually_remove_tags(text) if normalize_config.remove_tags else text

    extra_allowed_chars = [] if normalize_config.remove_tags else ["<", ">", "_"]

    def scrub(text: str) -> str:
        return "".join(char for char in text if char in charset + extra_allowed_chars)

    def final_fixes(text: str) -> str:
        return scrub(remove_tags(replace(text)))

    match normalize_config.normalize_level:
        case NormalizeLevel.IDENTITY:
            return remove_tags(replace(transcript))
        case NormalizeLevel.SCRUB:
            return final_fixes(transcript)
        case NormalizeLevel.ASCII:
            transcript_ = unidecode(transcript)
            return final_fixes(transcript_)
        case NormalizeLevel.DIGIT_TO_WORD:
            transcript_ = normalize_numbers(unidecode(transcript), charset)
            return final_fixes(transcript_)
        case NormalizeLevel.LOWERCASE:
            transcript_ = lowercase_normalize(
                transcript, quiet=False, charset=charset, scrub=False
            )

            if normalize_config.standardize_text:
                transcript_ = standardize_text(transcript_)

            if not transcript_:
                warnings.warn(f"Transcript normalization for {transcript=} returned ''")
                warnings.warn(
                    "Either normalization failed, or the original transcript was empty"
                )
            return final_fixes(transcript_)
        case _:
            raise ValueError(
                f"{NormalizeConfig.normalize_level=} is an unrecognized value "
                "of NormalizeLevel"
            )


@beartype
def standardize_text(text: str) -> str:
    """
    Standardize text. This function applies the Whisper normalization rules
    and the train-time normalizer on the text in order to minimize non-semantic
    text differences.


    Parameters
    ----------
    text
        string containing un-standardized text

    Returns
    -------
    standard_text
        string containing standardized text
    """
    standardizer = EnglishTextNormalizer()
    return standardizer(text)
