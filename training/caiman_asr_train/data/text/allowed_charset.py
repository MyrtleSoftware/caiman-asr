import string

from beartype import beartype
from beartype.typing import List, Set
from icu import Locale, LocaleData
from text_unidecode import unidecode

_ALLOWED_PUNCT = "'<>"
BANNED_PUNCT = "".join(x for x in string.punctuation if x not in _ALLOWED_PUNCT)


@beartype
def get_allowed_charset(langs: List[str], lowercase_only: bool = False) -> List[str]:
    """
    Generate a list of allowed characters for all languages in langs.
    Charset will include common punctuation and the characters for
    each language as derived using ICU (get_lang_charset).
    """

    allowed_charset = set(" ',.?0123456789%$¢€£-")

    for lang in langs:
        allowed_charset |= get_lang_charset(lang, lowercase_only)

    if lowercase_only:
        allowed_charset -= set(BANNED_PUNCT)
        allowed_charset |= set(_ALLOWED_PUNCT)

    return sorted((allowed_charset))


@beartype
def get_lang_charset(lang_code: str, lowercase_only: bool = False) -> Set[str]:
    """
    Generates a charset from a lang code using ICU's
    locale exemplar and it's upper case
    """

    assert is_supported(lang_code), f"Unknown/Unsupported language code: {lang_code}"

    locale_data = LocaleData(lang_code)

    # Get language-specific characters
    # getExemplarSet has 2 args and returns a UnicodeSet object.
    # The first arg is only relevant if using the UnicodeSet object
    # for pattern matching etc.
    # The second argument fetches subset of chars for the language/locale as follows:
    # 0 - Main chars for the locale
    # 1 - Auxiliary chars (foreign chars that might be used;
    #     example ú, á in case of English)
    # 2 - Index characters
    # 3 - Punctuation

    chars = set(locale_data.getExemplarSet(0, 0))
    if not lowercase_only:
        chars |= set(ch.upper() for ch in chars)

    return chars


@beartype
def is_supported(lang_code: str) -> bool:
    """Get a list of all locale IDs ICU supports"""

    available_locales = Locale.getAvailableLocales()

    return lang_code in available_locales


@beartype
def iso_unidecode(text: str, charset: List[str], ignore_case: bool = True) -> str:
    """Return text with chars not in the provided charset transliterated to ASCII

    Args:
        text: The input text to transliterate.
        charset: The list of allowed characters.
        ignore_case: If True, use lowercase characters.
    """

    # Angle quotes are commonly used in languages like French and Spanish.
    # These need to be preserved so as to not be transliterated to "<<" and ">>"
    #
    # Normalization cleaners will strip quotes for all languages.
    # Built once as a set (not a list) so the per-character membership check
    # below is O(1) instead of a linear scan repeated for every character.
    allowed = set(charset) | {"«", "»"}

    if ignore_case:
        return "".join(
            char if char.lower() in allowed else unidecode(char) for char in text
        )

    return "".join(char if char in allowed else unidecode(char) for char in text)
