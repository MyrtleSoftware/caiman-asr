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
Modified to add support for time and slight tweaks to _expand_number
"""

import re

import inflect

# Extend list of values after decillion
inflect.mill = [
    " ",
    " thousand",
    " million",
    " billion",
    " trillion",
    " quadrillion",
    " quintillion",
    " sextillion",
    " septillion",
    " octillion",
    " nonillion",
    " decillion",
    " undecillion",
    " duodecillion",
    " tredecillion",
    " quattuordecillion",
    " quindecillion",
    " hexdecillion",
    " septendecillion",
    " octodecillion",
    " novemdecillion",
    " vigintillion",
]

_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")
_time_re = re.compile(r"([0-9]{1,2}):([0-9]{2})")
_million_dollars_re = re.compile(r"\$[0-9]+([.][0-9]+)? million")
_billion_dollars_re = re.compile(r"\$[0-9]+([.][0-9]+)? billion")
_trillion_dollars_re = re.compile(r"\$[0-9]+([.][0-9]+)? trillion")
_thousand_dollars_re = re.compile(r"\$[0-9]+([.][0-9]+)? thousand")


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_decimal_point(m):
    return m.group(1).replace(".", " point ")


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    if int(m.group(0)[0]) == 0:
        return _inflect.number_to_words(m.group(0), andword="", group=1)
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(
                num, andword="", zero="oh", group=2
            ).replace(", ", " ")
    # Add check for number phones and other large numbers
    elif num > 1000000000 and num % 10000 != 0:
        return _inflect.number_to_words(num, andword="", group=1)
    else:
        return _inflect.number_to_words(num, andword="")


def _expand_time(m):
    mins = int(m.group(2))
    if mins == 0:
        return _inflect.number_to_words(m.group(1))
    return " ".join(
        [_inflect.number_to_words(m.group(1)), _inflect.number_to_words(m.group(2))]
    )


def _expand_million_dollars(m):
    return m.group(0)[1:] + " dollars"


def _expand_billion_dollars(m):
    return m.group(0)[1:] + " dollars"


def _expand_trillion_dollars(m):
    return m.group(0)[1:] + " dollars"


def _expand_thousand_dollars(m):
    return m.group(0)[1:] + " dollars"


def _pre_norm_num_expressions(text):
    """
    Clean up common numerical expressions before applying the normalization rules.
    """
    # add space between number and AM/PM
    text = re.sub(r"(?<=\d)(AM|PM)", r" \1", text, flags=re.IGNORECASE)

    # first address case 1-5 -> one to five
    text = re.sub(r"(?<=\d)-(?=\d)", " to ", text)

    # keep remaining dashes as "minus" because scrub will remove it
    text = re.sub(r"-(\d+)", r"minus \1", text)

    # keep hours and minutes separate because scrub will remove it
    text = re.sub(r"(?<=\d):(?=\d)", " ", text)
    return text


def _post_norm_num_expressions(text, charset: list[str]):
    """
    Post-normalization cleaning of numerical expressions.
    This is required since the lowercase_normalize function
    will replace the hyphens with empty strings if the hyphen
    is not in the charset.
    """
    # replace compound numbers hyphen
    if "-" not in charset:
        pattern = (
            r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
            r"-\b(one|two|three|four|five|six|seven|eight|nine)\b"
        )
        text = re.sub(pattern, r"\1 \2", text)

        # replace hyphen in cardinals
        card_pattern = (
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
            r"seventeen|eighteen|nineteen|twenty|"
            r"thirty|forty|fifty|sixty|seventy|eighty|ninety)"
            r"(?:-(one|two|three|four|five|six|seven|eight|nine))?"
            r"(?:-(half|third|fourth|quarter|fifth|sixth|seventh|"
            r"eighth|ninth|tenth|eleventh|twelfth|thirds|fourths|"
            r"quarters|fifths|sixths|sevenths|eighths|ninths|tenths|"
            r"elevenths|twelfths))?\b"
        )
        text = re.sub(
            card_pattern,
            lambda m: f"{m.group(1)} {m.group(2) or ''} {m.group(3) or ''}".strip(),
            text,
        )

    return text


def normalize_numbers(text, charset: list[str]):
    text = _pre_norm_num_expressions(text)
    text = re.sub(_million_dollars_re, _expand_million_dollars, text)
    text = re.sub(_billion_dollars_re, _expand_billion_dollars, text)
    text = re.sub(_trillion_dollars_re, _expand_trillion_dollars, text)
    text = re.sub(_thousand_dollars_re, _expand_thousand_dollars, text)
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    text = re.sub(_time_re, _expand_time, text)
    text = _post_norm_num_expressions(text, charset=charset)
    return text
