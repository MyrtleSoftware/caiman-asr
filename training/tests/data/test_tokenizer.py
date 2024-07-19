#! /usr/bin/env python

import unittest
from unittest.mock import patch

from caiman_asr_train.data.tokenizer import Tokenizer


class testTokenizer(unittest.TestCase):
    @patch("sentencepiece.SentencePieceProcessor")
    def test_input_spm_foo(self, mocked):
        tok = Tokenizer(labels=list("foobar"), sentpiece_model="foo")
        assert tok.charset == ["f", "o", "o", "b", "a", "r"]
        assert tok.sampling == 0.0

    def test_real_spm(self):
        pass


if __name__ == "__main__":
    unittest.main()
