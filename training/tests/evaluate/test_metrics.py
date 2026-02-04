#!/usr/bin/env python3
import random

import pytest
from hypothesis import given
from hypothesis.strategies import text
from levenshtein_rs import levenshtein_list as levenshtein

from caiman_asr_train.evaluate.error_rates import ErrorRate
from caiman_asr_train.evaluate.metrics import (
    calculate_keyword_wers,
    calculate_precision_recall_f1,
    count_keyword_stats,
    punctuation_error_rate,
    word_error_rate,
)


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


@pytest.mark.parametrize(
    "hypotheses, references, keywords, normalize, expected_tp, expected_fp, expected_fn",
    [
        (
            ["hello world", "hello mars"],
            ["hello world", "hello mars"],
            ["hello"],
            False,
            2,
            0,
            0,
        ),
        (
            ["hello march", "hello mars"],
            ["hello mars", "hello mars"],
            ["mars"],
            False,
            1,
            0,
            1,
        ),
        (
            ["hello Mars", "hello mars"],
            ["hello Mars", "hello Mars"],
            ["Mars"],
            False,
            1,
            0,
            1,
        ),
        (
            ["hello world", "hello mars"],
            ["hello world", "hello mars"],
            ["hello"],
            True,
            2,
            0,
            0,
        ),
        (
            ["hello march", "hello mars"],
            ["hello mars", "hello mars"],
            ["mars"],
            True,
            1,
            0,
            1,
        ),
        (
            ["hello Mars", "hello mars"],
            ["hello Mars", "hello Mars"],
            ["Mars"],
            True,
            2,
            0,
            0,
        ),
        (
            [
                "this is caiman asr and this is caiman",
                "this is caiman asr and this is cool",
            ],
            [
                "this is caiman asr and this is cool",
                " this is caiman asr and this is cool",
            ],
            ["caiman"],
            False,
            2,
            1,
            0,
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["Schuyler", "pathophysiology"],
            False,
            0,
            0,
            0,
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["Schuyler", "pathophysiology"],
            True,
            2,
            0,
            0,
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathology? Schuyler Schuyler"],
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["Schuyler", "pathophysiology"],
            True,
            1,
            2,
            1,
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathology? Schuyler Schuyler"],
            ["hello Dr. Schuyler, do you enjoy pathology?"],
            ["Schuyler", "pathophysiology"],
            True,
            1,
            2,
            0,
        ),
        (
            ["hello my name is Grayson Schuyler", "hello my name is Grayson School"],
            ["hello my name is Grayson Schuyler", "hello my name is Grayson Schuyler"],
            ["Grayson", "Schuyler"],
            True,
            3,
            0,
            1,
        ),
        (
            ["hello my name is Grayson Schuyler", "hello my name is Grayson School"],
            ["hello my name is Grayson Schuyler", "hello my name is Grayson Schuyler"],
            [
                "Grayson Schuyler"
            ],  # Doesn't currently support phrases (multi-word keywords)
            True,
            0,
            0,
            0,
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathology? Schuyler Schuyler"],
            ["hello Dr. Schuyler. Schuyler, do you enjoy pathophysiology?"],
            ["Schuyler", "pathophysiology"],
            True,
            1,
            2,
            2,
        ),
    ],
)
def test_count_keyword_stats(
    hypotheses, references, keywords, normalize, expected_tp, expected_fp, expected_fn
):
    tp, fp, fn = count_keyword_stats(
        hypotheses, references, keywords, normalize=normalize
    )
    assert (tp, fp, fn) == (
        expected_tp,
        expected_fp,
        expected_fn,
    ), "Keyword stats calculation is incorrect"


@pytest.mark.parametrize(
    "hypotheses, refs, keywords, normalize, expected_prec, expected_recall, expected_f1",
    [
        (
            ["hello world", "hello mars"],
            ["hello world", "hello mars"],
            ["hello"],
            False,
            1.0,
            1.0,
            1.0,
        ),
        (
            ["hello march", "hello mars"],
            ["hello mars", "hello mars"],
            ["mars"],
            False,
            1.0,
            0.5,
            0.6666666666666666,
        ),
        (
            ["hello Mars", "hello mars"],
            ["hello Mars", "hello Mars"],
            ["Mars"],
            False,
            1.0,
            0.5,
            0.6666666666666666,
        ),
        (
            ["hello world", "hello mars"],
            ["hello world", "hello mars"],
            ["hello"],
            True,
            1.0,
            1.0,
            1.0,
        ),
        (
            ["hello march", "hello mars"],
            ["hello mars", "hello mars"],
            ["mars"],
            True,
            1.0,
            0.5,
            0.6666666666666666,
        ),
        (
            ["hello Mars", "hello mars"],
            ["hello Mars", "hello Mars"],
            ["Mars"],
            True,
            1.0,
            1.0,
            1.0,
        ),
        (
            [
                "this is caiman asr and this is caiman",
                "this is caiman asr and this is cool",
            ],
            [
                "this is caiman asr and this is cool",
                " this is caiman asr and this is cool",
            ],
            ["caiman"],
            False,
            0.6666666666666666,
            1.0,
            0.8,
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["Schuyler", "pathophysiology"],
            False,
            0.0,
            0.0,
            0.0,
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["Schuyler", "pathophysiology"],
            True,
            1.0,
            1.0,
            1.0,
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathology? Schuyler Schuyler"],
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["Schuyler", "pathophysiology"],
            True,
            0.3333333333333333,
            0.5,
            0.4,
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathology? Schuyler Schuyler"],
            ["hello Dr. Schuyler, do you enjoy pathology?"],
            ["Schuyler", "pathophysiology"],
            True,
            0.3333333333333333,
            1.0,
            0.5,
        ),
    ],
)
def test_calculate_precision_recall_f1(
    hypotheses,
    refs,
    keywords,
    normalize,
    expected_prec,
    expected_recall,
    expected_f1,
):
    precision, recall, f1 = calculate_precision_recall_f1(
        hypotheses, refs, keywords, normalize
    )
    assert (precision, recall, f1) == (
        expected_prec,
        expected_recall,
        expected_f1,
    ), "Precision, recall, and F1 score calculation is incorrect"


@pytest.mark.parametrize(
    "hypotheses, refs, keywords, standardize, expected_wers",
    [
        (
            ["hello world", "hello mars"],
            ["hello world", "hello mars"],
            ["hello"],
            False,
            (0.0, None),
        ),
        (
            ["hello world", "hello mars"],
            ["hello world", "hello mars"],
            ["mars"],
            False,
            (0.0, 0.0),
        ),
        (
            ["hello Mars", "hello mars"],
            ["hello Mars", "hello Mars"],
            ["Mars"],
            False,
            (0.25, None),
        ),
        (
            ["hello Mars", "hello mars"],
            ["hello Mars", "hello Mars"],
            ["Mars"],
            True,
            (0.0, None),
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["Schuyler", "pathophysiology"],
            False,
            (None, 0),
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["Schuyler", "pathophysiology"],
            True,
            (0.0, None),
        ),
        (
            [
                "hello Dr. Schuyler, do you enjoy pathophysiology?",
                "I like pathology",
                "My name is Scooter",
            ],
            [
                "hello Dr. Schuyler, do you enjoy pathophysiology?",
                "I like pathophysiology!",
                "My name is Scooter",
            ],
            ["Schuyler", "pathophysiology"],
            True,
            (0.1, 0.0),
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathology? Schuyler Schuyler"],
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["Schuyler", "pathophysiology"],
            True,
            (3 / 7, None),
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathology? Schuyler Schuyler"],
            ["hello Dr. Schuyler, do you enjoy pathology?"],
            ["Schuyler", "pathophysiology"],
            True,
            (2 / 7, None),
        ),
        (
            ["there are no keywords in this sentence"],
            ["there are no keywords in this sentence"],
            ["Schuyler", "pathophysiology"],
            True,
            (None, 0.0),
        ),
        (
            [
                "CAIMAN ASR is the best!",
                "ASR stands for Automatic Speech Recognition",
                "Myrtle developed CAIMAN ASR",
                "Myrtle is based in Cambridge",
            ],
            [
                "CAIMAN ASR is the best!",
                "ASR stands for Automatic Speech Recognition",
                "Myrtle developed CAIMAN ASR",
                "Myrtle is based in Cambridge",
            ],
            ["CAIMAN", "ASR"],
            True,
            (0.0, 0.0),
        ),
        (
            [
                "Cayman ASR is the best!",
                "A S R stands for Automatic Speech Recognition",
                "Murt developed Cayman ASR",
                "Murt is based in Cambridge",
            ],
            [
                "CAIMAN ASR is the best!",
                "ASR stands for Automatic Speech Recognition",
                "Myrtle developed CAIMAN ASR",
                "Myrtle is based in Cambridge",
            ],
            ["CAIMAN", "ASR"],
            True,
            (0.4, 0.2),
        ),
    ],
)
def test_calculate_keyword_wers(hypotheses, refs, keywords, standardize, expected_wers):
    b_wer, u_wer = calculate_keyword_wers(hypotheses, refs, keywords, standardize)
    assert (b_wer, u_wer) == expected_wers, "Keyword WER calculation is incorrect"
