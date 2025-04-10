#!/usr/bin/env python3
import random

import pytest
from hypothesis import given
from hypothesis.strategies import text
from levenshtein_rs import levenshtein_list as levenshtein

from caiman_asr_train.evaluate.error_rates import ErrorRate
from caiman_asr_train.evaluate.metrics import punctuation_error_rate, word_error_rate


@given(text(), text())
def test_levenshtein1(a, b):
    assert levenshtein(list(a), list(b)) == levenshtein_ref(a, b)
    assert levenshtein(a.split(), b.split()) == levenshtein_ref(a.split(), b.split())


def test_levenshtein2():
    assert (
        levenshtein("Mary had a little lamb".split(), "Mary had a little lamb".split())
        == 0
    )
    assert levenshtein("I have a pet dog".split(), "You have a pet cat".split()) == 2
    assert levenshtein("one two three".split(), "two three".split()) == 1
    assert levenshtein("one two three".split(), "two three one".split()) == 2


def levenshtein_ref(a, b):
    """Calculates the Levenshtein distance between two sequences."""

    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


@pytest.mark.parametrize(
    "hypotheses, references, standardize, expected_wer",
    [
        (["hello world"], ["hello world"], True, (0.0, 0, 2)),
        (["hello world"], ["hi everyone"], True, (1.0, 2, 2)),
        ([], [], True, (float("inf"), 0, 0)),
        (["hello world"], ["hello new world"], True, (1 / 3, 1, 3)),
        (["good morning earth"], ["good morning mars good morning"], True, (0.6, 3, 5)),
    ],
)
def test_word_error_rate(hypotheses, references, standardize, expected_wer):
    wer, scores, words = word_error_rate(
        hypotheses, references, standardize=standardize, error_rate=ErrorRate.WORD
    )
    assert (wer, scores, words) == expected_wer, "WER calculation is incorrect"


def test_wer_unequal_lengths():
    references = ["hello", "mars"]
    hypotheses = ["hello"]
    with pytest.raises(ValueError):
        word_error_rate(
            hypotheses, references, standardize=True, error_rate=ErrorRate.WORD
        )


def test_long_utterance():
    str1 = " ".join(random.choice(["a", "b"]) for _ in range(700))
    str2 = " ".join(random.choice(["a", "b"]) for _ in range(700))
    assert levenshtein(list(str1), list(str2)) == levenshtein_ref(str1, str2)
    assert levenshtein(str1.split(), str2.split()) == levenshtein_ref(
        str1.split(), str2.split()
    )


def test_mer():
    assert word_error_rate(
        ["One two four一二四"],
        ["One two three一二三"],
        standardize=False,
        error_rate=ErrorRate.MIXTURE,
    ) == (1 / 3, 2, 6)


@pytest.mark.parametrize(
    "hypotheses, references, expected_per",
    [
        (
            ["Brief. I pray! For you.", ""],
            ["Brief. <EOS> I pray now for you;", ""],
            (2 / 3, 3),
        ),
        (
            ["Hey. 'She said; 'How are you, ok?'"],
            ["Hey! She said, 'How are you; ok?'"],
            (0.0, 4),
        ),
        (
            ["say hello, there you!", "That's the same!"],
            ["say hello there you.", "That's not the same!"],
            (1 / 3, 3),
        ),
        (
            ["say hello there you", "That's the same"],
            ["say hello there you", "That's not the same"],
            (0.0, 0),
        ),
    ],
)
def test_punctuation_error_rate(hypotheses, references, expected_per):
    per, pun_symbols = punctuation_error_rate(hypotheses, references)
    assert (per, pun_symbols) == expected_per, "PER calculation is incorrect"
