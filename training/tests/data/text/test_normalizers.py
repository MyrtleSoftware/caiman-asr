#!/usr/bin/env python3
import string

import pytest
from hypothesis import given
from hypothesis.strategies import text

from caiman_asr_train.data.text.normalizers import (
    punctuation_normalize,
    select_and_normalize,
)
from caiman_asr_train.data.text.preprocess import norm_and_tokenize
from caiman_asr_train.setup.text_normalization import (
    NormalizeConfig,
    NormalizeLevel,
    get_normalize_config,
)


def test_digit_normalize():
    """Compare to test_normalize_file.py"""
    allowed_chars = list(string.ascii_letters + " '.,!?-")
    allowed_chars_default = list(string.ascii_letters + " '")

    def norm(x):
        return select_and_normalize(
            x,
            allowed_chars,
            NormalizeConfig(NormalizeLevel.DIGIT_TO_WORD, [], False, [], False),
        )

    def norm_default(x):
        return select_and_normalize(
            x,
            allowed_chars_default,
            NormalizeConfig(NormalizeLevel.LOWERCASE, [], False, [], False),
        )

    assert norm("Testing, 1, 2, 3.") == "Testing, one, two, three."
    assert norm("你好 means hello") == "Ni Hao  means hello"
    assert norm("être is a french verb") == "etre is a french verb"
    assert norm("   hello   there    ") == "   hello   there    "
    assert (
        norm("<words> <inside> <tags> <like_these> aren't removed")
        == "<words> <inside> <tags> <like_these> aren't removed"
    )
    assert norm("Mr. and Mrs. Lincoln") == "Mr. and Mrs. Lincoln"
    assert (
        norm("I earned $101 in 1999!")
        == "I earned one hundred one dollars in nineteen ninety-nine!"
    )
    assert (
        norm("chiefly between 1845 and 1849")
        == "chiefly between eighteen forty-five and eighteen forty-nine"
    )
    assert (
        norm_default("chiefly between 1845 and 1849")
        == "chiefly between eighteen forty five and eighteen forty nine"
    )
    assert norm(("set the alarm for 3:16AM")) == "set the alarm for three sixteen AM"
    assert norm(("set the alarm for 3:16 AM")) == "set the alarm for three sixteen AM"
    assert (
        norm(("calculate -37.0 plus 38.35"))
        == "calculate minus thirty-seven point zero plus thirty-eight point thirty-five"
    )
    assert (
        norm_default(("set the alarm for 3:16AM"))
        == "set the alarm for three sixteen am"
    )
    assert norm(("two-fifths and three-quarters")) == "two-fifths and three-quarters"
    assert (
        norm_default(("two-fifths and three-quarters"))
        == "two fifths and three quarters"
    )
    assert norm(("about 1-5")) == "about one to five"
    assert norm_default(("about 1-5")) == "about one to five"
    assert norm_default(("about one-five")) == "about one five"
    assert norm_default(("  word-word     ")) == "wordword"
    assert norm_default(("foo:bar")) == "foo bar"
    assert norm(("foo:bar")) == "foobar"


def test_lowercase_normalize():
    """Compare to test_normalize_file.py with the default normalization"""
    allowed_chars = list(string.ascii_letters + " '")

    def norm(x):
        return select_and_normalize(
            x,
            allowed_chars,
            NormalizeConfig(NormalizeLevel.LOWERCASE, [], False, [], False),
        )

    def norm_and_standard(x):
        return select_and_normalize(
            x,
            allowed_chars,
            NormalizeConfig(NormalizeLevel.LOWERCASE, [], False, [], True),
        )

    assert (
        norm(("set the alarm for 3:16AM aeon"))
        == "set the alarm for three sixteen am aeon"
    )
    assert (
        norm(("calculate -37.0 plus 38.35"))
        == "calculate minus thirty seven point zero plus thirty eight point thirty five"
    )
    assert (
        norm_and_standard(("centralised aeon defence marvellously"))
        == "centralized eon defense marvelously"
    )
    assert norm_and_standard("Testing, 1, 2, 3.") == "testing one two three"
    assert (
        norm_and_standard("<words> <inside> <tags> <like_these> are removed")
        == "are removed"
    )


@pytest.mark.parametrize(
    "normalizer,remove_tags,expected",
    [
        (
            NormalizeLevel.IDENTITY,
            False,
            "<silence> Mr. Smith visited the café @ 123rd Street.",
        ),
        (NormalizeLevel.SCRUB, False, "<silence> Mr Smith visited the caf  rd Street"),
        (NormalizeLevel.ASCII, False, "<silence> Mr Smith visited the cafe  rd Street"),
        (
            NormalizeLevel.DIGIT_TO_WORD,
            False,
            "<silence> Mr Smith visited the cafe  one hundred and twenty-third Street",
        ),
        (
            NormalizeLevel.LOWERCASE,
            False,
            "<silence> mister smith visited the cafe"
            " at one hundred and twenty-third street",
        ),
        (
            NormalizeLevel.IDENTITY,
            True,
            " Mr. Smith visited the café @ 123rd Street.",
        ),
        (NormalizeLevel.SCRUB, True, " Mr Smith visited the caf  rd Street"),
        (NormalizeLevel.ASCII, True, " Mr Smith visited the cafe  rd Street"),
        (
            NormalizeLevel.DIGIT_TO_WORD,
            True,
            " Mr Smith visited the cafe  one hundred and twenty-third Street",
        ),
        (
            NormalizeLevel.LOWERCASE,
            True,
            " mister smith visited the cafe at one hundred and twenty-third street",
        ),
    ],
)
def test_variants(normalizer, remove_tags, expected):
    charset = list(string.ascii_letters + " -")
    text = "<silence> Mr. Smith visited the café @ 123rd Street."
    normalize_config = NormalizeConfig(normalizer, [], remove_tags, [], False)
    result = norm_and_tokenize(
        text, tokenizer=None, normalize_config=normalize_config, charset=charset
    )
    assert result == expected


@pytest.mark.parametrize("normalizer", NormalizeLevel)
def test_implemented(normalizer):
    """Make sure norm_and_tokenize can handle all NormalizeLevels,
    including ones added later"""
    normalize_config = NormalizeConfig(normalizer, [], False, [], False)
    norm_and_tokenize(
        "foo",
        tokenizer=None,
        normalize_config=normalize_config,
        charset=["a", "b", "c"],
    )


@given(input_text=text())
@pytest.mark.parametrize("remove_tags", [True, False])
@pytest.mark.parametrize(
    "normalizer",
    [
        NormalizeLevel.SCRUB,
        NormalizeLevel.ASCII,
        NormalizeLevel.DIGIT_TO_WORD,
        NormalizeLevel.LOWERCASE,
    ],
)
def test_no_forbidden_chars(input_text, default_charset, normalizer, remove_tags):
    normalize_config = NormalizeConfig(normalizer, [], remove_tags, [], False)
    result = norm_and_tokenize(
        input_text,
        tokenizer=None,
        normalize_config=normalize_config,
        charset=default_charset,
    )
    extra_allowed_chars = [] if remove_tags else ["<", ">", "_"]
    for char in result:
        assert char in default_charset + extra_allowed_chars


@pytest.mark.parametrize(
    "normalizer,expected",
    [
        ("identity", "Blue is a color, red is a color, yellow is   also a color"),
        ("scrub", "Blue is a color, red is a color, yellow is   also a color"),
        ("ascii", "Blue is a color, red is a color, yellow is   also a color"),
        ("digit_to_word", "Blue is a color, red is a color, yellow is   also a color"),
        ("lowercase", "blue is a color, red is a color, yellow is   also a color"),
    ],
)
def test_replacements(normalizer, expected):
    charset = list(string.ascii_letters + " ,")
    replacements = [
        {"old": ";", "new": ","},
        {"old": "-", "new": " "},
        {"old": "colour", "new": "color"},
    ]
    text = "Blue is a colour; red is a colour; yellow is---also a colour"
    normalize_config = get_normalize_config(normalizer, replacements, False, [], False)
    result = norm_and_tokenize(
        text, tokenizer=None, normalize_config=normalize_config, charset=charset
    )
    assert result == expected


def test_tag_removal(default_charset):
    def norm(x):
        return select_and_normalize(
            x,
            default_charset,
            NormalizeConfig(NormalizeLevel.LOWERCASE, [], True, [], False),
        )

    assert norm("<tags> <like_these> are removed") == "  are removed"
    assert norm("<inaudible>") == ""
    assert norm("to-") == "to"
    assert norm("<affirmative> <silence>") == " "
    assert norm("<affirmative> <sw> <inaudible>") == "  "


@pytest.mark.parametrize(
    "text, norm_text",
    [
        ('Hey! \'She said: "How are you; ok?"', "Hey. 'She said. How are you, ok?"),
        ("-hello! <EOS>", " hello. <EOS>"),
    ],
)
def test_punctuation_normalize(text, norm_text):
    ret_text = punctuation_normalize(text)
    assert ret_text == norm_text
