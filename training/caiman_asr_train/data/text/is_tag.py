import re

from beartype import beartype


def actually_remove_tags(text):
    """
    Remove tags or codes inside angled brackets
    >>> actually_remove_tags('testing <unk> <inaudible> one <foreign_word> three')
    'testing   one  three'
    """
    return re.sub("<[a-zA-Z_]+>", "", text)


@beartype
def is_tag(word: str) -> bool:
    return actually_remove_tags(word).strip() == "" and word.strip() != ""
