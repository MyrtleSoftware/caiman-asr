#!/usr/bin/env python3

# Originally from: https://github.com/openai/whisper/blob/main/whisper/normalizers/basic.py
# Additions/amendments by Myrtle.ai


import re
import unicodedata

import regex

from caiman_asr_train.data.text.whisper_basic_normalizer import (
    remove_symbols_and_diacritics,
)


def remove_symbols(s: str, keep: str = ""):
    """
    Replace any other markers, symbols, punctuation with a space, keeping diacritics
    """
    # MSP = Mark, Symbol, Punctuation
    # https://www.unicode.org/reports/tr44/tr44-34.html#General_Category_Values
    # NFKD and NFKC normalise nbsp to breaking space. Check requirement for CJK langs
    return "".join(
        c if c in keep else (" " if unicodedata.category(c)[0] in "MSP" else c)
        for c in unicodedata.normalize("NFKC", s)
    )


class BasicTextNormalizer:
    def __init__(
        self,
        preserve_case: bool = False,
        preserve_punctuation: bool = False,
        remove_diacritics: bool = False,
        split_letters: bool = False,
    ):
        self.preserve_case = preserve_case
        self.preserve_punctuation = preserve_punctuation
        self.clean = (
            remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        )
        self.split_letters = split_letters

    def __call__(self, s: str):
        if not self.preserve_case:
            s = s.lower()

        s = re.sub(r"[«»]", "", s)  # remove angle quotes
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis

        # Lines 51-71 are not in the original whisper normalizer
        # This is based on the Whisper English Normalizer, but modified for other langs
        if not self.preserve_punctuation:
            # Numbers in most European langs use , for decimal and . as a separator
            # remove any period between digits
            s = re.sub(r"(\d)\.(\d)", r"\1\2", s)
            # remove any period not following a digit
            s = re.sub(r"([^0-9])\.(\s|$)", r"\1 \2", s)
            # remove comma not followed by a digit
            s = re.sub(r",([^0-9]|$)", r" \1", s)

            s = self.clean(s, keep=".,-%$¢€£'")

            # remove any currency symbols not followed by digits or not following digits
            s = re.sub(r"(^(([$¢€£](\s?\d+))|((\d+\s?)[$¢€£])))", r" \1 ", s)
            # remove any % not preceded by numbers
            s = re.sub(r"([^0-9])%", r"\1 ", s)

            # remove any hyphens not within words. Hyphens within words can alter
            # the context for some European languages like Hungarian, Maltese
            s = re.sub(r"(\s)-\s", r"\1 ", s)

            if not self.preserve_case:
                # for some non-standard symbols like "𝖧" which str.lower() wouldn't convert
                s = s.lower()

        if self.split_letters:
            s = " ".join(regex.findall(r"\X", s, regex.U))

        s = re.sub(
            r"\s+", " ", s
        )  # replace any successive whitespace characters with a space

        # This part is not in the original whisper normalizer
        # This is based on the Whisper English Normalizer
        if not re.search(r"\w", s):
            return ""

        return s.strip()
