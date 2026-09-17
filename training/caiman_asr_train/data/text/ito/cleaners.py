# Copyright (c) 2017 Keith Ito
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2023, Myrtle Software Limited, www.myrtle.ai. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""from https://github.com/keithito/tacotron
Modified to add punctuation removal

Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the
"cleaners" hyperparameter. Some cleaners are English-specific. You'll typically want to use:
    1. "english_cleaners" for English text
    2. "transliteration_cleaners" for non-English text that can be transliterated to
        ASCII using the Text-Unidecode library (https://pypi.org/project/text-unidecode/)
    3. "basic_cleaners" if you do not want to transliterate (in this case, you should
        also update the symbols in symbols.py to match your data).

"""

import re

from beartype import beartype
from beartype.typing import Optional
from text_unidecode import unidecode

from caiman_asr_train.data.text.allowed_charset import iso_unidecode
from caiman_asr_train.data.text.ito.numbers import normalize_numbers

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# Regular expression for angle quotes/chevron:
_angle_quotes_re = re.compile(r"[«»]")
_strip_angle_quote_tokens_re = re.compile(r"«[^»]*»")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "missus"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def strip_tokens_in_angle_quotes(text):
    return re.sub(_strip_angle_quote_tokens_re, " ", text)


def strip_angle_quotes(text):
    return re.sub(_angle_quotes_re, " ", text)


def expand_punctuation(text, charset):
    """
    Expands pronounceable punctuation characters if they are not in the charset.
    """

    if "&" not in charset:
        text = re.sub(r"&", " and ", text)

    if "#" not in charset:
        text = re.sub(r"#", " number ", text)

    if "+" not in charset:
        text = re.sub(r"\+", " plus ", text)

    if "%" not in charset:
        text = re.sub(r"%", " percent ", text)

    if "@" not in charset:
        text = re.sub(r"@", " at ", text)

    if "=" not in charset:
        text = re.sub(r"=", " equal ", text)

    if ":" not in charset:
        text = re.sub(r":", " ", text)

    return text


def english_cleaners(text, charset, table=None):
    """Pipeline for English text, including number and abbreviation expansion."""
    text = strip_tokens_in_angle_quotes(text)
    text = strip_angle_quotes(text)
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = normalize_numbers(text, charset)
    text = expand_abbreviations(text)
    if table is not None:
        text = text.translate(table)
    text = expand_punctuation(text, charset)
    text = collapse_whitespace(text)
    return text


@beartype
def other_cleaners(
    text: str, charset: list[str], table: Optional[dict[int, int]] = None
) -> str:
    """Pipeline for text in languages other than English"""
    text = strip_angle_quotes(text)
    text = iso_unidecode(text, charset=charset)
    text = lowercase(text)
    if table is not None:
        text = text.translate(table)
    text = collapse_whitespace(text)
    return text
