#! /usr/bin/env python3

import pytest

from caiman_asr_train.evaluate.punctuation_error_rate import (
    punctuation_error_rate_function,
)


@pytest.fixture
def ref_hyp():
    return ["Hi, dear! Nice to see you. What's"], ["Hi dear! Nice to see you! What's?"]


@pytest.fixture
def ref_hyp2():
    return ["Hi, dear! Nice to see you. What's", "hello, there!"], [
        "Hi dear! Nice to see you! What's?",
        "hey there!",
    ]


@pytest.mark.parametrize(
    "punctuation_marks, expected_per",
    [
        (
            [".", ",", "!", "?"],
            (0.75, 4),
        ),
        (
            ["."],
            (1.0, 1),
        ),
        (
            [","],
            (1.0, 1),
        ),
        (
            ["!"],
            (0.5, 2),
        ),
        (
            ["?"],
            (1.0, 1),
        ),
    ],
)
def test_punctuation_error_rate_function(ref_hyp, punctuation_marks, expected_per):
    per, num_oper = punctuation_error_rate_function(
        ref_hyp[0], ref_hyp[1], punctuation_marks
    )
    assert abs(per - expected_per[0]) < 1e-6, "PER calculation is incorrect"
    assert num_oper == expected_per[1], "Number of operations is incorrect"


def test_punctuation_error_rate_function2(ref_hyp2):
    punctuation_marks = [".", ",", "?"]
    per, num_oper = punctuation_error_rate_function(
        ref_hyp2[0], ref_hyp2[1], punctuation_marks
    )
    assert abs(per - 4 / 4) < 1e-6, "PER calculation is incorrect"
    assert num_oper == 4, "Number of operations is incorrect"
