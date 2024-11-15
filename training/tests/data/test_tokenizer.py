#! /usr/bin/env python

import unittest
from unittest.mock import patch

import pytest

from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.data.unk_handling import UnkHandling
from caiman_asr_train.rnnt.config import get_tokenizer_conf


class testTokenizer(unittest.TestCase):
    @patch("sentencepiece.SentencePieceProcessor")
    def test_input_spm_foo(self, mocked):
        tok = Tokenizer(
            labels=list("foobar"),
            sentpiece_model="foo",
            sampling=0.0,
            unk_handling=UnkHandling.FAIL,
        )
        assert tok.charset == ["f", "o", "o", "b", "a", "r"]
        assert tok.sampling == 0.0

    def test_real_spm(self):
        pass


def test_tokenizer_unk(tokenizer):
    assert "" == tokenizer.sentpiece.decode(0)
    assert "c" == tokenizer.sentpiece.decode(15)
    assert "a" == tokenizer.sentpiece.decode(5)
    assert "t" == tokenizer.sentpiece.decode(4)
    # 0 detokenizes differently when part of a list
    assert " ⁇ cat" == tokenizer.sentpiece.decode([0, 15, 5, 4])
    assert " ⁇ " == tokenizer.sentpiece.decode([0])


def test_eos():
    alphas = list(" abcdefghijklmnopqrstuvwxyz'")

    config = {
        "labels": alphas,
        "sampling": 0.0,
    }

    wo_eos_config = {
        **config,
        "sentpiece_model": "/workspace/training/tests/test_data/librispeech29.model",
    }

    wo_eos = Tokenizer(**get_tokenizer_conf({"tokenizer": wo_eos_config}))

    wi_eos_config = {
        **config,
        "sentpiece_model": "/workspace/training/tests/test_data/librispeech30.eos.model",
    }

    wi_eos = Tokenizer(**get_tokenizer_conf({"tokenizer": wi_eos_config}))

    assert wi_eos.charset == wo_eos.charset

    assert wo_eos.num_labels == len(alphas) + 1  # + unk
    assert wi_eos.num_labels == len(alphas) + 2  # + unk + eos

    # Should decode EOS
    wi_eos.tokenize("hi bob <EOS>")
    # Should not decode EOS
    with pytest.raises(ValueError):
        wo_eos.tokenize("hi bob <EOS>")


if __name__ == "__main__":
    unittest.main()
