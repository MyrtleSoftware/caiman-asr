#!/usr/bin/env python3
import re
import warnings
from functools import lru_cache

import inflect
from beartype import beartype
from beartype.typing import Optional

from caiman_asr_train.data.text.allowed_charset import (
    get_allowed_charset,
    iso_unidecode,
)
from caiman_asr_train.data.text.basic_normalizer import BasicTextNormalizer
from caiman_asr_train.data.text.is_tag import actually_remove_tags
from caiman_asr_train.data.text.ito import punctuation_map
from caiman_asr_train.data.text.ito.cleaners import english_cleaners, other_cleaners
from caiman_asr_train.data.text.symbols_mask import (
    generate_symbols_mask,
    mask_user_symbols,
    unmask_user_symbols,
)
from caiman_asr_train.data.text.whisper_text_normalizer import EnglishTextNormalizer
from caiman_asr_train.setup.text_normalization import NormalizeConfig, Replacement


@beartype
def _apply_replacements(text: str, replacements: list[Replacement]) -> str:
    for r in replacements:
        text = text.replace(r.old, r.new)
    return text


@beartype
def _scrub(text: str, charset: list[str]) -> str:
    """
    Remove any characters from text that are not in the allowed charset.

    The characters are replaced with a space.

    Args:
        text (str): The input text to scrub.
        charset (list[str]): The list of allowed characters.

    Returns:
        str: The scrubbed text.
    """
    allowed = set(charset)
    replaced = "".join(c if c in allowed else " " for c in text)
    return " ".join(replaced.split())


@beartype
def language_normalize(
    transcript: str,
    charset: list[str],
    config: NormalizeConfig,
    lang_code: str,
    mask: dict[str, str] | None = None,
) -> str:
    """Normalize text according to language-specific rules.

    Pipeline:
      1. iso_unidecode: get acceptable characters per language
            (e.g. "é" -> "e", "ß" -> "ss")
      2. standardize_text: Whisper-based normalizer (case/punct/contractions/
            filler words), applied only if config.standardize_text
      3. _apply_replacements: user-defined string substitutions from config
      4. actually_remove_tags: strip non-user-defined <tags>
      5. _scrub: space-replace any chars outside the charset

    Parameters
    ----------
    transcript
        Raw text to normalize.
    charset
        Allowed characters (from the tokenizer for the target language).
    config
        Normalization settings: replacements, standardize_text,
        preserve_case, preserve_punctuation.
    lang_code
        BCP-47 language code (e.g. "en", "fr").
    mask
        Optional dictionary mapping user symbols to masked placeholders.
    """

    text = mask_user_symbols(transcript, mask)
    text = iso_unidecode(text, charset=charset, ignore_case=True)
    if config.standardize_text:
        text = standardize_text(
            text,
            lang_code,
            preserve_case=config.preserve_case,
            preserve_punctuation=config.preserve_punctuation,
            expand_contractions=True,
        )

    if not text:
        warnings.warn(
            f"Transcript normalization for {transcript=} returned ''."
            " Either normalization failed, or the original transcript was empty"
        )

    text = _apply_replacements(text, config.replacements)
    text = actually_remove_tags(text)
    text = _scrub(text, charset)
    return unmask_user_symbols(text, mask)


@beartype
def punctuation_normalize(text: str) -> str:
    """Normalize punctuation.

    This is used during the metric calculation.

    This function simplifies a few punctuation signs and is used during
    the punctuation error rate calculation.

    The following substitutions are made:

    `"` -> `` strip double quote
    `;` -> `,` semi-colon to comma
    `:` -> `.` colon to period
    `!` -> `.` exclamation mark to period
    `-` -> ` ` hyphen to space
    NB that the hyphen may be separating two words.

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
    s,
    charset: list[str],
    lang_code: str,
    quiet: bool = False,
    scrub: bool = True,
) -> str:
    """Normalizes string.

    Example:
    >>> import string
    >>> lowercase_normalize('call me at 8:00 pm!', \
            charset=list(string.ascii_lowercase+' '), lang_code='en')
    'call me at eight zero zero pm'
    """
    # When called from the training pipeline, scrub should be False,
    # since the scrubbing should happen after the replacements.
    # When called from elsewhere, scrub=True for backwards compatibility.
    punct_map = punctuation_map(charset) if scrub else None

    try:
        cleaner = english_cleaners if lang_code == "en" else other_cleaners
        text = cleaner(s, charset, punct_map).strip()
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


@lru_cache(maxsize=None)
@beartype
def _get_standardizer(
    lang_code: str,
    preserve_case: bool,
    preserve_punctuation: bool,
    expand_contractions: bool,
) -> BasicTextNormalizer | EnglishTextNormalizer:
    # Cached: EnglishTextNormalizer's EnglishSpellingNormalizer re-reads and
    # compiles a ~1700-entry mapping on every construction, so building one of
    # these per call (rather than once per distinct parameter combination) is
    # a significant, easily-avoided cost on the per-utterance normalization path.
    if lang_code == "en":
        return EnglishTextNormalizer(
            preserve_case=preserve_case,
            preserve_punctuation=preserve_punctuation,
            expand_contractions=expand_contractions,
        )
    return BasicTextNormalizer(
        preserve_case=preserve_case,
        preserve_punctuation=preserve_punctuation,
    )


@beartype
def standardize_text(
    text: str,
    lang_code: str,
    preserve_case: bool = False,
    preserve_punctuation: bool = False,
    expand_contractions: bool = True,
) -> str:
    """
    Standardize text. This function applies the Whisper normalization rules
    and the train-time normalizer on the text in order to minimize non-semantic
    text differences.


    Parameters
    ----------
    text
        string containing un-standardized text
    lang_code
        language code to use for standardization
    preserve_case
        whether to preserve case (default: False)
    preserve_punctuation
        whether to preserve punctuation (default: False)
    expand_contractions
        whether to expand contractions (default: True)

    Returns
    -------
    standard_text
        string containing standardized text
    """

    standardizer = _get_standardizer(
        lang_code, preserve_case, preserve_punctuation, expand_contractions
    )
    return standardizer(text)


@beartype
class MemoizedLowercase:
    def __init__(
        self,
        user_symbols: list[str],
        max_cache_size: Optional[int] = None,
    ):
        """
        A caching lowercase normalizer.

        This class is required because applying lru_cache directly to a method
        is a footgun: https://rednafi.com/python/lru_cache_on_methods/

        Args:
            user_symbols: List of user symbols to mask during lowercasing.
            max_cache_size: Maximum size of the LRU cache. If None unbounded.
        """
        self.mask = generate_symbols_mask(user_symbols)
        self.lowercase = lru_cache(maxsize=max_cache_size)(self._lowercase)

    def _lowercase(self, text: str, lang_code: str) -> Optional[str]:
        masked = mask_user_symbols(text, self.mask)

        charset = get_allowed_charset([lang_code], lowercase_only=True)
        normalized = lowercase_normalize(masked, charset, lang_code)
        normalized = standardize_text(
            normalized, lang_code, preserve_case=False, preserve_punctuation=False
        )

        if normalized == masked:
            # Normalizing was a no-op: the input already was clean lowercase,
            # so signal this to PrefixTokenize to force the nopnc prefix.
            return None

        return unmask_user_symbols(normalized, self.mask)

    def __call__(self, text: str, lang_code: str) -> Optional[str]:
        return self.lowercase(text, lang_code)

    def normalize(self, text: str, lang_code: str) -> str:
        """Apply lowercasing and standardization, always returning a str.

        Use this instead of __call__ when a str result is always required
        (e.g. for WER computation). __call__ is intentionally Optional[str]
        so PrefixTokenize can detect when normalizing was a no-op.
        """
        result = self.lowercase(text, lang_code)
        return text if result is None else result


@beartype
def remove_user_symbols(text: str, user_symbols: list[str]) -> str:
    """
    Replace user symbols.

    Parameters
    ----------
    text
        string containing user symbols
    user_symbols
        list of user symbols to conceal

    Returns
    -------
    text
        string with all user symbols removed
    """
    # regex containing all user symbols
    user_symbols_regex = "|".join(re.escape(symbol) for symbol in user_symbols)
    text = re.sub(user_symbols_regex, "", text)
    # collapse whitespace left behind by a removed symbol, but leave trailing
    # whitespace untouched since callers strip it separately
    return re.sub(r" {2,}(?=\S)", " ", text)
