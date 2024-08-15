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


def test_tokenizer_unk(tokenizer):
    assert "" == tokenizer.sentpiece.decode(0)
    assert "c" == tokenizer.sentpiece.decode(15)
    assert "a" == tokenizer.sentpiece.decode(5)
    assert "t" == tokenizer.sentpiece.decode(4)
    # 0 detokenizes differently when part of a list
    assert " ⁇ cat" == tokenizer.sentpiece.decode([0, 15, 5, 4])
    assert " ⁇ " == tokenizer.sentpiece.decode([0])


if __name__ == "__main__":
    unittest.main()
