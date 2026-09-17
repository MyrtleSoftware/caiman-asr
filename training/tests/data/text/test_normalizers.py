#!/usr/bin/env python3
import string

import pytest

from caiman_asr_train.data.text.allowed_charset import (
    get_allowed_charset,
    get_lang_charset,
)
from caiman_asr_train.data.text.normalizers import (
    language_normalize,
    punctuation_normalize,
    remove_user_symbols,
)
from caiman_asr_train.data.text.preprocess import norm_and_tokenize
from caiman_asr_train.setup.text_normalization import (
    NormalizeConfig,
    get_normalize_config,
    normalize_config_from_config_data,
)


@pytest.mark.parametrize(
    "text, symbols, expected",
    [
        ("hello <EOS> <*>", ["<EOS>", "<*>"], "hello  "),
        ("<hi> <[unk]> hi", ["<EOS>", "<[unk]>"], "<hi> hi"),
    ],
)
def test_remove_user_symbols(text, symbols, expected):
    assert remove_user_symbols(text, symbols) == expected


def test_digit_normalize():
    """Compare to test_normalize_file.py"""
    # allowed_chars = list(string.ascii_letters + " '.,!?-$0123456789")
    allowed_chars = get_allowed_charset(["en"], lowercase_only=False)

    def norm(x):
        return language_normalize(
            x,
            allowed_chars,
            NormalizeConfig([], [], {}, True, True, True),
            lang_code="en",
            mask={"aaaaa": "<pnc>"},
        )

    assert norm("Testing, 1, 2, 3.") == "Testing, 1, 2, 3."
    assert norm("你好 means hello") == "Ni Hao means hello"
    assert norm("être is a french verb") == "etre is a french verb"
    assert norm("   hello   there    ") == "hello there"
    assert (
        norm("<words> <inside> <tags> <like_these> aren't removed") == "are not removed"
    )
    assert norm("Mr. and Mrs. Lincoln") == "Mister and Missus Lincoln"
    assert norm("I earned $101 in 1999!") == "I earned $101 in 1999"
    assert norm("chiefly between 1845 and 1849") == "chiefly between 1845 and 1849"
    assert (
        norm(("set the alarm for 3:16AM not 4:16 AM"))
        == "set the alarm for 3 16AM not 4 16 AM"
    )
    assert norm(("")) == ""
    assert norm(("<pnc> D&D AM/FM")) == "<pnc> D and D AM FM"
    assert norm(("calculate -37.0 plus 38.35")) == "calculate -37.0 plus 38.35"
    assert norm(("two-fifths and three-quarters")) == "two-fifths and three-quarters"
    assert norm(("about 1-5")) == "about 1-5"
    assert norm(("foo:bar")) == "foo bar"


def test_lowercase_normalize():
    """Compare to test_normalize_file.py with the default normalization"""
    allowed_chars = list(string.ascii_letters + " '.,!?-0123456789")

    def norm(x):
        return language_normalize(
            x,
            allowed_chars,
            NormalizeConfig([], [], {}, True, True, True),
            lang_code="en",
        )

    assert norm(("set the alarm for 3:16AM aeon")) == "set the alarm for 3 16AM eon"
    assert norm(("calculate -37.0 plus 38.35")) == "calculate -37.0 plus 38.35"
    assert (
        norm(("centralised aeon defence marvellously"))
        == "centralized eon defense marvelously"
    )
    assert norm("Testing, 1, 2, 3.") == "Testing, 1, 2, 3."
    assert norm("<words> <inside> <tags> <like_these> are removed") == "are removed"


def test_variants():
    # Test for norm_and_tokenize function.
    # Tags are always removed (via the Whisper normalizer's bracket
    # stripping), there's no config knob for this any more.
    charset = list(string.ascii_letters + " -<>@0123456789")
    text = "<silence> Mr. Smith visited the café @ 123rd Street."
    normalize_config = NormalizeConfig([], [], {}, True, True, True)
    result = norm_and_tokenize(
        text,
        tokenizer=None,
        normalize_config=normalize_config,
        charset=charset,
        lang_code="en",
    )
    assert result == "Mister Smith visited the cafe at 123rd Street"


@pytest.mark.parametrize(
    "charset, text, expected",
    [
        (
            list(string.ascii_letters + " ,"),
            "Blue is a color",
            "Blue is a color",
        ),
        (
            list(string.ascii_lowercase + " ,"),
            "Blue is a color 11.1",
            "lue is a color",
        ),
        (
            list(get_lang_charset("en") | set(" ")),
            "Blue is a color",
            "Blue is a color",
        ),
        (
            list(get_lang_charset("de") | set(" ")),
            "Esta es una Prüfung rápida",
            "Esta es una Prüfung rapida",
        ),
    ],
)
def test_digit_to_word(charset, text, expected):
    replacements = [
        {"old": ";", "new": ","},
    ]
    normalize_config = get_normalize_config(replacements, [], {}, True, True, True)
    result = norm_and_tokenize(
        text,
        tokenizer=None,
        normalize_config=normalize_config,
        charset=charset,
        lang_code="en",
    )
    assert result == expected


def test_tag_removal(default_charset):
    def norm(x):
        return language_normalize(
            x,
            default_charset,
            NormalizeConfig([], [], {}, True, True, True),
            lang_code="en",
        )

    assert norm("<tags> <like_these> are removed") == "are removed"
    assert norm("<inaudible>") == ""
    assert norm("to-") == "to"
    assert norm("<affirmative> <silence>") == ""
    assert norm("<affirmative> <sw> <inaudible>") == ""


def test_standardize_text_disabled_preserves_filler_words(default_charset):
    """
    Validation runs with standardize_text=False so that a filler-only
    reference doesn't collapse to '' and get silently dropped by the
    min_tokens cutset filter (see language_normalize).
    """

    def norm(x, standardize_text):
        return language_normalize(
            x,
            default_charset,
            NormalizeConfig([], [], {}, standardize_text, True, True),
            lang_code="en",
        )

    assert norm("Hmm.", standardize_text=True) == ""
    assert norm("Hmm.", standardize_text=False) != ""


def test_standardize_text_default_differs_by_split():
    """Train defaults to standardize_text=True, val to False, when the YAML
    doesn't set the key explicitly (val is in fact forbidden from setting it,
    enforced in rnnt/config.py)."""
    config_data = {}
    train_config = normalize_config_from_config_data(config_data, [], {}, mode="train")
    val_config = normalize_config_from_config_data(config_data, [], {}, mode="val")
    assert train_config.standardize_text is True
    assert val_config.standardize_text is False


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


@pytest.mark.parametrize(
    "text, expected, charset, normalize_config",
    [
        # lowercase
        (
            "I am British and this is how I spell sanitise! co op and co op but co-op",
            "I am British and this is how I spell sanitize Company op and company op "
            "but company op",
            list(string.ascii_letters + " "),
            NormalizeConfig([], [], {}, True, True, True),
        ),
        # + add punc to charset
        (
            "I am British and this is how I spell sanitise!",
            "I am British and this is how I spell sanitize!",
            list(string.ascii_letters + " !?.,-"),
            NormalizeConfig([], [], {}, True, True, True),
        ),
        (
            "UHHHH HOW MANY WORDS ARE THERE IN THIS SENTENCE?",
            "HOW MANY WORDS ARE THERE IN THIS SENTENCE?",
            list(string.ascii_letters + " !?.,-"),
            NormalizeConfig([], [], {}, True, True, True),
        ),
        (
            "Dr. you are my saviour",
            "Doctor you are my savior",
            list(string.ascii_letters + " !?.,-"),
            NormalizeConfig([], [], {}, True, True, True),
        ),
        (
            "One two 3 4 5 SIX SEVEN 8 9 Ten!",
            "One two 3 4 5 SIX SEVEN 8 9 Ten!",
            list(string.ascii_letters + " !?.,-0123456789"),
            NormalizeConfig([], [], {}, True, True, True),
        ),
        (
            "One two 3 4 5 SIX SEVEN 8 9 Ten!",
            "One two SIX SEVEN Ten!",
            list(string.ascii_letters + " !?.,-"),
            NormalizeConfig([], [], {}, True, True, True),
        ),
        (
            "No. Um, um, well",
            "No. Well",
            list(string.ascii_letters + " !?.,-"),
            NormalizeConfig([], [], {}, True, True, True),
        ),
        (
            "We represent individual, um. The fastest way is meetings",
            "We represent individual. The fastest way is meetings",
            list(string.ascii_letters + " !?.,-"),
            NormalizeConfig([], [], {}, True, True, True),
        ),
        (
            '"We represent individual, um. The fastest way is meetings"',
            "We represent individual. The fastest way is meetings",
            list(string.ascii_letters + " !?.,-"),
            NormalizeConfig([], [], {}, True, True, True),
        ),
        (
            '"We represent individual, um. The fastest way is meetings"',
            '"We represent individual. The fastest way is meetings"',
            list(string.ascii_letters + ' !?.,-"'),
            NormalizeConfig([], [], {}, True, True, True),
        ),
        # lowercase + standardize
        (
            '"We represent individual, um. The fastest way is meetings"',
            "we represent individual the fastest way is meetings",
            list(string.ascii_letters + " "),
            NormalizeConfig([], [], {}, True, False, False),
        ),
        (
            "No. Um, um, well",
            "no well",
            list(string.ascii_letters + " "),
            NormalizeConfig([], [], {}, True, False, False),
        ),
        (
            (
                "Dr. umm Smith mmhmm umm, yes please, MMM. "
                "Prof. um Smith and umm prof. umm, yes please."
            ),
            "doctor smith yes please professor smith and professor yes please",
            list(string.ascii_letters + " "),
            NormalizeConfig([], [], {}, True, False, False),
        ),
        (
            "UH MM-HMM UH HUH-HUH MMMM. HMMM?!",
            "",
            list(string.ascii_letters + " "),
            NormalizeConfig([], [], {}, True, False, False),
        ),
    ],
)
def test_language_normalize(text, expected, charset, normalize_config):
    result = language_normalize(
        text,
        charset=charset,
        config=normalize_config,
        lang_code="en",
    )
    assert result == expected


@pytest.mark.parametrize(
    "text, expected, charset, normalize_config, lang_code",
    [
        (
            "I am British «habladuría política» and this is how I spell sanitise!",
            "I am British and this is how I spell sanitize",
            list(string.ascii_letters + " <>"),
            NormalizeConfig([], [], {}, True, True, True),
            "en",
        ),
        (
            "I am British «habladuría política» and this is how I spell sanitise!",
            "I am British habladuria politica and this is how I spell sanitise",
            list(string.ascii_letters + " <>"),
            NormalizeConfig([], [], {}, True, True, True),
            "fr",
        ),
        # lowercase-only charset
        (
            "I am British «habladuría política» and this is how I spell sanitise!",
            "am ritish and this is how spell sanitize",
            list(string.ascii_lowercase + " <>"),
            NormalizeConfig([], [], {}, True, True, True),
            "en",
        ),
    ],
)
def test_language_normalize_angle_quotes(
    text, expected, charset, normalize_config, lang_code
):
    result = language_normalize(
        text,
        charset=charset,
        config=normalize_config,
        lang_code=lang_code,
    )
    assert result == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # angle bracket tags
        ("<silence> hello world", "hello world"),
        ("hello <unk> world", "hello world"),
        ("speak <foreign_word> clearly", "speak clearly"),
        ("<|inaudible|> testing one two", "testing one two"),
        # multiple tags
        ("<silence> foo <unk> bar", "foo bar"),
        # tag in the middle and at the end
        ("before <silence> after", "before after"),
        ("hello world <silence>", "hello world"),
        # square bracket tags
        ("[laughter] hello world", "hello world"),
        ("[noise] speaking [applause]", "speaking"),
        # mixed angle and square bracket tags
        ("<silence> hello [laughter] world <unk>", "hello world"),
    ],
)
def test_language_normalize_tag_removal(text, expected):
    charset = list(string.ascii_letters + " ")
    config = NormalizeConfig([], [], {}, True, True, True)
    result = language_normalize(text, charset=charset, config=config, lang_code="en")
    assert result == expected


def test_language_normalize_general():
    charset = list(string.ascii_letters + " '.,!?-")

    def norm(x):
        return language_normalize(
            x,
            charset,
            NormalizeConfig([], [], {}, True, True, True),
            lang_code="en",
        )

    # basic pass-through
    assert norm("hello world") == "hello world"
    # non-ASCII transliteration via iso_unidecode
    assert norm("café au lait") == "cafe au lait"
    assert norm("über fast") == "uber fast"
    # contraction expansion
    assert norm("she's here") == "she is here"
    assert norm("aren't they") == "are not they"
    # chars outside charset are silently dropped
    assert norm("foo:bar") == "foo bar"
    # chars outside charset become a space
    assert norm("foo:bar") == "foo bar"
    # leading/trailing whitespace is stripped
    assert norm("  hello world  ") == "hello world"
