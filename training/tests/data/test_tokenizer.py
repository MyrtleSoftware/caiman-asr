#! /usr/bin/env python

import unittest
from unittest.mock import patch

from caiman_asr_train.data.tokenizer import Tokenizer


class testTokenizer(unittest.TestCase):
    def test_input_no_spm(self):
        tok = Tokenizer(labels="bar")
        assert tok.charset == "bar"
        assert tok.use_sentpiece is False
        assert tok.num_labels == 3

        expected_label2ind = {"b": 0, "a": 1, "r": 2}
        assert set(tok.label2ind.keys()) == set(expected_label2ind.keys())
        for key, val in tok.label2ind.items():
            assert expected_label2ind[key] == val

        assert tok.tokenize("bar") == [0, 1, 2]

    @patch("sentencepiece.SentencePieceProcessor")
    def test_input_spm_foo(self, mocked):
        tok = Tokenizer(labels="foobar", sentpiece_model="foo")
        assert tok.charset == "foobar"
        assert tok.use_sentpiece is True
        assert tok.sampling == 0.0

    def test_real_spm(self):
        pass


if __name__ == "__main__":
    unittest.main()
