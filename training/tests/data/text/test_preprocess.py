#! /usr/bin/nev python

import pytest

from caiman_asr_train.data.text.preprocess import mask_user_symbols


@pytest.mark.parametrize(
    "text, expected, _dict",
    [
        ("hello <EOS>", "hello mask", {"mask": "<EOS>"}),
        (
            "hello <[unk]> <unk>",
            "hello john mary",
            {"mary": "<unk>", "john": "<[unk]>"},
        ),
        ("hello <*>", "hello  ", {" ": "<*>"}),
        ("hello <S>", "hello <S>", None),
    ],
)
def test_mask_user_symbols(text, expected, _dict):
    assert mask_user_symbols(text, _dict) == expected
