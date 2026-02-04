import re

from beartype import beartype

_ANGLE_TAG = r"<[\\/|]?[a-zA-Z0-9_]+[\\/|]?>"
_SQUARE_TAG = r"\[[^\]]+\]"
_TAG_REGEX = re.compile(f"(?:{_ANGLE_TAG})|(?:{_SQUARE_TAG})")


def actually_remove_tags(text):
    """
    Remove tags or codes inside angled brackets
    >>> actually_remove_tags('testing <unk> <|inaudible|> one <foreign_word> three')
    'testing   one  three'
    """
    return _TAG_REGEX.sub("", text)


@beartype
def is_tag(word: str) -> bool:
    return actually_remove_tags(word).strip() == "" and word.strip() != ""


@beartype
def is_word_tag(word: str) -> bool:
    """
    Check if the word is a tag or a code inside angled brackets
    >>> is_word_tag('<unk>')
    True
    >>> is_word_tag('testing')
    False
    """
    return _TAG_REGEX.match(word) is not None
