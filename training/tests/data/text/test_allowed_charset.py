import pytest

from caiman_asr_train.data.text.allowed_charset import (
    get_allowed_charset,
    get_lang_charset,
    iso_unidecode,
)


@pytest.mark.parametrize(
    "lang_code, expected_chars",
    [
        ("es", ["a", "A", "ñ", "Ñ", "á", "Á"]),
        ("de", ["a", "A", "ü", "Ü", "ö", "Ö", "ß"]),
        ("fr", ["a", "A", "é", "É", "è", "È", "ç", "Ç"]),
        ("en", ["a", "A", "z", "Z"]),
    ],
)
def test_get_lang_charset_contains_expected(lang_code, expected_chars):
    """Test that charset contains expected language-specific characters"""
    charset = get_lang_charset(lang_code)
    for char in expected_chars:
        assert char in charset, f"Expected '{char}' in {lang_code} charset"


@pytest.mark.parametrize(
    "lang_code, expected_chars",
    [
        ("es", ["A", "Ñ", "Á", "<"]),
        ("de", ["A", "Ü", "Ö", "."]),
        ("fr", ["A", "É", "È", "Ç", "?"]),
        ("en", ["A", "Z", "0", "8", "?", ">"]),
    ],
)
def test_get_lang_charset_does_not_contain_uppercase(lang_code, expected_chars):
    """Test that charset contains expected language-specific characters"""
    charset = get_lang_charset(lang_code, lowercase_only=True)
    for char in expected_chars:
        assert char not in charset, f"Unexpected '{char}' in {lang_code} charset"


@pytest.mark.parametrize(
    "langs, must_include",
    [
        (["es", "de"], ["ñ", "ü", "ö", "A", "m"]),
        (["fr"], ["é", "è", "ç", "0", "9", "'"]),
        (["en", "fr"], ["a", "z", "0", "9", ",", "."]),
    ],
)
def test_get_allowed_charset_contains_expected(langs, must_include):
    """Test that allowed charset contains expected characters"""
    charset = get_allowed_charset(langs)
    for char in must_include:
        assert char in charset, f"Expected '{char}' in charset for {langs}"


@pytest.mark.parametrize(
    "langs, must_not_include",
    [
        (["es", "de"], ["<", ">"]),
        (["en"], ["<", ">"]),
    ],
)
def test_get_allowed_charset_does_not_contain_banned_punct(langs, must_not_include):
    """Test that allowed charset contains expected characters"""
    charset = get_allowed_charset(langs)
    for char in must_not_include:
        assert char not in charset, f"Unexpected '{char}' in charset for {langs}"


@pytest.mark.parametrize(
    "langs, must_not_include",
    [
        (["es", "de"], [".", "?", "!"]),
        (["en"], [".", "?", "!"]),
    ],
)
def test_get_allowed_charset_lowercase_does_not_contain_banned_punct(
    langs, must_not_include
):
    """Test that allowed charset contains expected characters"""
    charset = get_allowed_charset(langs, lowercase_only=True)
    for char in must_not_include:
        assert char not in charset, f"Unexpected '{char}' in charset for {langs}"


@pytest.mark.parametrize(
    "langs, must_include",
    [
        (["es", "de"], ["ñ", "ü", "ö", "<", "'"]),
        (["en"], ["a", "z", ">", "'"]),
    ],
)
def test_get_allowed_charset_lowercase_contains_expected(langs, must_include):
    """Test that allowed charset contains expected characters"""
    charset = get_allowed_charset(langs, lowercase_only=True)
    for char in must_include:
        assert char in charset, f"Expected '{char}' in charset for {langs}"


@pytest.mark.parametrize(
    "langs",
    [["en"]],
)
def test_get_allowed_charset_for_en(langs):
    """Test that charset always includes numbers and basic punctuation"""
    charset = get_allowed_charset(langs)
    digits = "0123456789"
    for char in digits:
        assert char in charset, f"Expected char in charset '{char}'"


@pytest.mark.parametrize(
    "text, langs, expected",
    [
        # Characters in charset should be preserved
        ("hello", ["en"], "hello"),
        ("café", ["es"], "café"),
        ("Müller", ["de"], "Müller"),
        ("señor", ["es"], "señor"),
        ("Esta es una Prüfung", ["es", "de"], "Esta es una Prüfung"),
        # Characters not in charset should be transliterated
        ("café", ["en"], "cafe"),
        ("Москва", ["en"], "Moskva"),
        ("Αθήνα", ["en"], "Athena"),
        ("Esta es una Prüfung", ["en"], "Esta es una Prufung"),
        ("Café München — ¡rápido!", ["es", "de"], "Café München -- !rápido!"),
        ("Café München — «rápido»", ["es", "de"], "Café München -- «rápido»"),
        # Numbers and punctuation always preserved
        ("Test 123", ["en"], "Test 123"),
        ("Hello, world?", ["en"], "Hello, world?"),
        # Unicode punctuation converted
        ("Wait…", ["en"], "Wait..."),
        ('She said, "Hi"', ["en"], 'She said, "Hi"'),
    ],
)
def test_iso_unidecode(text, langs, expected):
    """Test iso_unidecode with various inputs"""
    charset = get_allowed_charset(langs)
    assert iso_unidecode(text, charset) == expected


@pytest.mark.parametrize(
    "text, langs, expected",
    [
        ("SEÑOR", ["es"], "SENOR"),
        ("Esta es una Prüfung", ["es", "de"], "Esta es una Prüfung"),
        ("café", ["en"], "cafe"),
        ("Москва", ["en"], "Moskva"),
        ("Αθήνα", ["en"], "Athena"),
        ("Café München — ¡rápido!", ["es", "de"], "Café München -- !rápido!"),
        ("Café München — «rápido»", ["es", "de"], "Café München -- «rápido»"),
    ],
)
def test_iso_unidecode_strict(text, langs, expected):
    """Test iso_unidecode with various inputs"""
    charset = get_allowed_charset(langs, lowercase_only=True)
    assert iso_unidecode(text, charset, ignore_case=False) == expected
