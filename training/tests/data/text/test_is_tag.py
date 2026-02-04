#!/usr/bin/env python3

import pytest

from caiman_asr_train.data.text.is_tag import actually_remove_tags, is_tag, is_word_tag


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "testing <unk> <|inaudible|> one <foreign_word> three",
            "testing   one  three",
        ),
        ("normal text without tags", "normal text without tags"),
        ("<tag>only tag</tag>", "only tag"),
        ("<|tag1|><tag2></tag3/>", ""),
        ("mixed <tag>with</tag> and <another_tag> text", "mixed with and  text"),
        ("<<<tag> is this a <\\tag>?", "<< is this a ?"),
        ("is this a <tag >?", "is this a <tag >?"),
        ("is this a [inaudible 00:03:10]?", "is this a ?"),
    ],
)
def test_actually_remove_tags(text, expected):
    assert actually_remove_tags(text) == expected


@pytest.mark.parametrize(
    "word, expected",
    [
        ("<unk>", True),
        ("<tag>", True),
        ("normal", False),
        ("", False),
        ("<foreign_word>", True),
        ("<|inaudible|>", True),
        ("not<tag>", False),
        ("<tag>not", False),
        ("<tag1><tag2/>", True),
    ],
)
def test_is_tag(word, expected):
    assert is_tag(word) == expected


@pytest.mark.parametrize(
    "word, expected",
    [
        ("<unk>", True),
        ("<tag>", True),
        ("normal", False),
        ("", False),
        ("<foreign_word>", True),
        ("<|inaudible|>", True),
        ("not<tag>", False),
        ("<tag>not", True),
        ("<tag1><tag2/>", True),
    ],
)
def test_is_word_tag(word, expected):
    assert is_word_tag(word) == expected
