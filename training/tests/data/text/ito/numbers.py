#!/usr/bin/env python3

import pytest

from caiman_asr_train.data.text.ito.numbers import (
    _post_norm_num_expressions,
    normalize_numbers,
)


@pytest.fixture
def charset():
    return list("abcdefghijklmnopqrstuvwxyz '")


@pytest.mark.parametrize("in_text, expected", [("five-thirds", "five  thirds")])
def test_normalize_numbers(in_text, expected, charset):
    returned = normalize_numbers(in_text, charset)
    assert returned == expected


@pytest.mark.parametrize("in_text, expected", [("five-thirds", "five  thirds")])
def test_post_norm_num_expressions(in_text, expected, charset):
    returned = _post_norm_num_expressions(in_text, charset)
    assert returned == expected
