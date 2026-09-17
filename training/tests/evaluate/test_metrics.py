#!/usr/bin/env python3
import random

import pytest
from hypothesis import given
from hypothesis.strategies import text
from rapidfuzz.distance.Levenshtein import distance as levenshtein

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
    "hypotheses, references, expected_wer",
    [
        (["hello world"], ["hello world"], (0.0, 0, 2)),
        (["hello world"], ["hi everyone"], (1.0, 2, 2)),
        ([], [], (float("inf"), 0, 0)),
        (["hello world"], ["hello new world"], (1 / 3, 1, 3)),
        (["good morning earth"], ["good morning mars good morning"], (0.6, 3, 5)),
    ],
)
def test_word_error_rate(hypotheses, references, expected_wer):
    wer_details = word_error_rate(
        hypotheses,
        references,
        error_rate=ErrorRate.WORD,
        lang_code="en",
    )
    wer, scores, words = wer_details.wer, wer_details.scores, wer_details.words
    assert (wer, scores, words) == expected_wer, "WER calculation is incorrect"


@pytest.mark.parametrize(
    (
        "hypotheses, references, lang_code, "
        "expected_wer, expected_hyp_norm, expected_ref_norm"
    ),
    [
        (
            ["hello world"],
            ["Hello World."],
            "en",
            (0.0, 0, 2),
            "hello world",
            "hello world",
        ),
        (
            ["Eugène, est le cousé germais des mille luis Buronov."],
            ["Eugène est le cousin germain d'Émile-Louis Burnouf."],
            "fr",
            (0.75, 6, 8),
            "eugène est le cousé germais des mille luis buronov",
            "eugène est le cousin germain d'émile louis burnouf",
        ),
        (
            ["Eugène Burnouf, est le cousé  germais des mille luis Buronov."],
            ["Eugène Burnouf  est le cousin germain d'Émile-Louis  Burnouf."],
            "fr",
            (2 / 3, 6, 9),
            "eugène burnouf est le cousé germais des mille luis buronov",
            "eugène burnouf est le cousin germain d'émile louis burnouf",
        ),
        ([], [], "en", (float("inf"), 0, 0), "", ""),
        (
            ["calificó de habladuría política y boberías a los rumores"],
            ["calificó de «habladuría política y boberías» a los rumores"],
            "es",
            (0.0, 0, 9),
            "calificó de habladuría política y boberías a los rumores",
            "calificó de habladuría política y boberías a los rumores",
        ),
    ],
)
def test_word_error_rate_standardization(
    hypotheses,
    references,
    lang_code,
    expected_wer,
    expected_hyp_norm,
    expected_ref_norm,
):
    wer_details = word_error_rate(
        hypotheses,
        references,
        error_rate=ErrorRate.WORD,
        lang_code=lang_code,
    )
    print(wer_details)
    wer, scores, words = wer_details.wer, wer_details.scores, wer_details.words
    assert (wer, scores, words) == expected_wer, "WER calculation is incorrect"
    assert (wer_details.norm_hyp, wer_details.norm_ref) == (
        expected_hyp_norm,
        expected_ref_norm,
    ), "WER standardization is incorrect"


@pytest.mark.parametrize(
    "hypotheses, references, expected_subs, expected_ins, expected_dels",
    [
        (["hello world"], ["hello world"], 0, 0, 0),
        (["a x c"], ["a b c"], 1, 0, 0),
        (["a b c d"], ["a b c"], 0, 1, 0),
        (["a b"], ["a b c"], 0, 0, 1),
        (["a x c d"], ["a b c"], 1, 1, 0),
        (["a x"], ["a b c"], 1, 0, 1),
        (["a b c d"], ["a x c"], 1, 1, 0),
        (["x y z", "a b"], ["a b c", "a b c"], 3, 0, 1),
    ],
)
def test_word_error_rate_details(
    hypotheses, references, expected_subs, expected_ins, expected_dels
):
    wer_details = word_error_rate(
        hypotheses,
        references,
        error_rate=ErrorRate.WORD,
        lang_code="en",
    )
    assert wer_details.substitutions == expected_subs
    assert wer_details.insertions == expected_ins
    assert wer_details.deletions == expected_dels
    assert wer_details.scores == expected_subs + expected_ins + expected_dels


def test_wer_unequal_lengths():
    references = ["hello", "mars"]
    hypotheses = ["hello"]
    with pytest.raises(ValueError):
        word_error_rate(
            hypotheses,
            references,
            error_rate=ErrorRate.WORD,
            lang_code="en",
        )


def test_long_utterance():
    str1 = " ".join(random.choice(["a", "b"]) for _ in range(700))
    str2 = " ".join(random.choice(["a", "b"]) for _ in range(700))
    assert levenshtein(list(str1), list(str2)) == levenshtein_ref(str1, str2)
    assert levenshtein(str1.split(), str2.split()) == levenshtein_ref(
        str1.split(), str2.split()
    )


def test_mer():
    wer_details = word_error_rate(
        ["One two four 一二四"],
        ["One two three 一二三"],
        error_rate=ErrorRate.MIXTURE,
        lang_code="en",
    )
    wer, scores, words = wer_details.wer, wer_details.scores, wer_details.words
    assert (wer, scores, words) == (1 / 3, 2, 6)


def test_mer2():
    """
    The "One一" is treated as single word, as there is not space between the words.
    """
    wer_details = word_error_rate(
        ["One一二四"],
        ["One一二三"],
        error_rate=ErrorRate.MIXTURE,
        lang_code="en",
    )
    wer, scores, words = wer_details.wer, wer_details.scores, wer_details.words
    assert (wer, scores, words) == (1 / 3, 1, 3)


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
    per_details = punctuation_error_rate(hypotheses, references)
    per, pun_symbols = per_details.per, per_details.punc_symbols
    assert (per, pun_symbols) == expected_per, "PER calculation is incorrect"


@pytest.mark.parametrize(
    "hypotheses, references, keywords, lang_code, "
    "expected_tp, expected_fp, expected_fn",
    [
        (
            ["hello world", "hello mars"],
            ["hello world", "hello mars"],
            ["hello"],
            "en",
            2,
            0,
            0,
        ),
        (
            ["hello march", "hello mars"],
            ["hello mars", "hello mars"],
            ["mars"],
            "en",
            1,
            0,
            1,
        ),
        (
            ["hello Mars", "hello mars"],
            ["hello Mars", "hello Mars"],
            ["Mars"],
            "en",
            2,
            0,
            0,
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["Schuyler", "pathophysiology"],
            "en",
            2,
            0,
            0,
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathology? Schuyler Schuyler"],
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["Schuyler", "pathophysiology"],
            "en",
            1,
            2,
            1,
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathology? Schuyler Schuyler"],
            ["hello Dr. Schuyler, do you enjoy pathology?"],
            ["Schuyler", "pathophysiology"],
            "en",
            1,
            2,
            0,
        ),
        (
            ["hello my name is Grayson Schuyler", "hello my name is Grayson School"],
            ["hello my name is Grayson Schuyler", "hello my name is Grayson Schuyler"],
            ["Grayson", "Schuyler"],
            "en",
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
            "en",
            0,
            0,
            0,
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathology? Schuyler Schuyler"],
            ["hello Dr. Schuyler. Schuyler, do you enjoy pathophysiology?"],
            ["Schuyler", "pathophysiology"],
            "en",
            1,
            2,
            2,
        ),
        (
            ["hyp", "", ""],
            ["", "ref", ""],  # Test nothrow on empty reference
            [],
            "en",
            0,
            0,
            0,
        ),
    ],
)
def test_count_keyword_stats(
    hypotheses,
    references,
    keywords,
    lang_code,
    expected_tp,
    expected_fp,
    expected_fn,
):
    tp, fp, fn = count_keyword_stats(hypotheses, references, keywords, lang_code)
    assert (tp, fp, fn) == (
        expected_tp,
        expected_fp,
        expected_fn,
    ), "Keyword stats calculation is incorrect"


@pytest.mark.parametrize(
    "hypotheses, refs, keywords, lang_code, "
    "expected_prec, expected_recall, expected_f1",
    [
        (
            ["hello world", "hello mars"],
            ["hello world", "hello mars"],
            ["hello"],
            "en",
            1.0,
            1.0,
            1.0,
        ),
        (
            ["hello march", "hello mars"],
            ["hello mars", "hello mars"],
            ["mars"],
            "en",
            1.0,
            0.5,
            0.6666666666666666,
        ),
        (
            ["hello Mars", "hello mars"],
            ["hello Mars", "hello Mars"],
            ["Mars"],
            "en",
            1.0,
            1.0,
            1.0,
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["Schuyler", "pathophysiology"],
            "en",
            1.0,
            1.0,
            1.0,
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathology? Schuyler Schuyler"],
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["Schuyler", "pathophysiology"],
            "en",
            0.3333333333333333,
            0.5,
            0.4,
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathology? Schuyler Schuyler"],
            ["hello Dr. Schuyler, do you enjoy pathology?"],
            ["Schuyler", "pathophysiology"],
            "en",
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
    lang_code,
    expected_prec,
    expected_recall,
    expected_f1,
):
    precision, recall, f1 = calculate_precision_recall_f1(
        hypotheses, refs, keywords, lang_code
    )
    assert (precision, recall, f1) == (
        expected_prec,
        expected_recall,
        expected_f1,
    ), "Precision, recall, and F1 score calculation is incorrect"


@pytest.mark.parametrize(
    "hypotheses, refs, keywords, lang_code, expected_wers",
    [
        (
            ["hello Mars", "hello mars"],
            ["hello Mars", "hello Mars"],
            ["Mars"],
            "en",
            (0.0, None),
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["Schuyler", "pathophysiology"],
            "en",
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
            "en",
            (0.1, 0.0),
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathology? Schuyler Schuyler"],
            ["hello Dr. Schuyler, do you enjoy pathophysiology?"],
            ["Schuyler", "pathophysiology"],
            "en",
            (3 / 7, None),
        ),
        (
            ["hello Dr. Schuyler, do you enjoy pathology? Schuyler Schuyler"],
            ["hello Dr. Schuyler, do you enjoy pathology?"],
            ["Schuyler", "pathophysiology"],
            "en",
            (2 / 7, None),
        ),
        (
            ["there are no keywords in this sentence"],
            ["there are no keywords in this sentence"],
            ["Schuyler", "pathophysiology"],
            "en",
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
            "en",
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
            "en",
            (0.4, 0.2),
        ),
    ],
)
def test_calculate_keyword_wers(hypotheses, refs, keywords, lang_code, expected_wers):
    b_wer, u_wer = calculate_keyword_wers(hypotheses, refs, keywords, lang_code)
    assert (b_wer, u_wer) == expected_wers, "Keyword WER calculation is incorrect"
