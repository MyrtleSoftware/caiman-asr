#!/usr/bin/env python3
import pytest

from caiman_asr_train.data.text.whisper_text_normalizer import EnglishTextNormalizer
from caiman_asr_train.evaluate.metrics import standardize_wer


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (("hello, lemme help"), ("hello let me help")),
        (("wanna see today's menu"), ("want to see today's menu")),
        (("I'd have what you'd have"), ("i would have what you would have")),
        (("utilize will be utilise"), ("utilize will be utilize")),
        (("Aren't <Silence> [inaudible] (audible)"), ("are not audible")),
        (("£ 20"), (" 20")),
        (
            ("hmm that's what we'll normalise in today's example"),
            ("that is what we will normalize in today's example"),
        ),
    ],
)
def test_whisper(test_input, expected):
    normalizer = EnglishTextNormalizer()
    returned = normalizer(test_input)
    assert expected == returned


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("hello, lemme help", "hello let me help"),
        ("wanna see today's menu", "want to see today's menu"),
        ("I'd have what you'd have", "i would have what you would have"),
        ("utilize will be utilise", "utilize will be utilize"),
        (
            "Aren't <Silence> [audible] (audible)",
            "are not audible audible",
        ),
        ("£ 20", "ps twenty"),
        ("$ 20", "twenty"),
        ("$20", "twenty dollars"),
        (
            "hmm that's what we'll standardise in today's example",
            "that is what we will standardize in today's example",
        ),
        ("Dr. Smith", "doctor smith"),
        ("$1.02", "one dollar two cents"),
        ("café", "cafe"),
        ("cats & dogs", "cats and dogs"),
        ("<tags> <will_disappear>", ""),
    ],
)
def test_standardize(test_input, expected):
    returned = standardize_wer(test_input)
    assert expected == returned
