import string

import pytest
from lhotse import SupervisionSegment

from caiman_asr_train.data.text.normalizers import MemoizedLowercase
from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.data.unk_handling import UnkHandling
from caiman_asr_train.lhotse.text import MemoizedNormalize, PrefixTokenize
from caiman_asr_train.setup.text_normalization import NormalizeConfig


@pytest.mark.parametrize(
    "input_text, lang_id, expected_output",
    [
        ("Hello, World!", "en", "hello world"),
        ("Καλημέρα σας! Πώς είστε;", "el", "kalemera sas pos eiste"),
        ("This is a test.", "en", "this is a test"),
        ("Mixed CASE and Punctuation!", "en", "mixed case and punctuation"),
    ],
)
def test_normalize(input_text, lang_id, expected_output):
    _config = NormalizeConfig(
        replacements=[],
        user_symbols=["<nopnc>"],
        user_symbols_dict={"nopnc": "<nopnc>"},
        standardize_text=True,
        preserve_case=False,
        preserve_punctuation=False,
    )
    _norm = MemoizedNormalize(
        config=_config, charset=list(string.ascii_lowercase) + [" ", "'"]
    )

    returned = _norm(input_text, lang_id)
    assert (
        returned == expected_output
    ), f"Expected '{expected_output}', got '{returned}'"


@pytest.mark.parametrize(
    "input_text, lang_id, expected_output",
    [
        ("Hello, World!", "en", "hello world"),
        ("Καλημέρα σας! Πώς είστε;", "el", "καλημέρα σας πώς είστε"),
        ("This is a test.", "en", "this is a test"),
        ("Mixed CASE and Punctuation!", "en", "mixed case and punctuation"),
    ],
)
def test_normalize_lowercase(input_text, lang_id, expected_output):
    _norm_lower = MemoizedLowercase(
        user_symbols=["<pnc>", "<nopnc>"],
    )

    returned = _norm_lower(input_text, lang_id)
    assert (
        returned == expected_output
    ), f"Expected '{expected_output}', got '{returned}'"


@pytest.mark.parametrize(
    "input_text, lang_id, expected_output, prob, tokens, prefix",
    [
        (
            "Hello, World.",
            "en",
            "Hello, World.",
            "1.0",
            [4, 32, 5, 13, 13, 9, 21, 4, 33, 9, 12, 13, 14, 24],
            [2],
        ),
        (
            "Hello, World.",
            "en",
            "Hello, World.",
            "0.0",
            [4, 11, 5, 13, 13, 9, 4, 18, 9, 12, 13, 14],
            [3],
        ),
        (
            "Guten Tag",
            "de",
            "Guten Tag",
            "0.0",
            [4, 26, 15, 6, 5, 19, 4, 6, 7, 26],
            [3],
        ),
    ],
)
def test_prefixtokenize(
    test_data_dir, input_text, lang_id, expected_output, prob, tokens, prefix
):
    _tokenizer = Tokenizer(
        labels=list(string.ascii_lowercase) + ["<EOS>", "<unk>", "<pnc>", "<nopnc>"],
        unk_handling=UnkHandling.FAIL,
        sentpiece_model=str(test_data_dir / "librispeech62.eos.pnc-tag.model"),
        sampling=0.0,
    )
    pref_tok = PrefixTokenize(
        tokenizer=_tokenizer,
        user_symbols={"pnc": "<pnc>", "nopnc": "<nopnc>", "EOS": "<EOS>"},
        pnc_prob=float(prob),
    )
    seg = SupervisionSegment(
        id="rec00001-sup00001",
        recording_id="rec00001",
        start=5.5,
        duration=3.0,
        channel=0,
        text=input_text,
        language=lang_id,
    )
    seg_text = pref_tok(seg).text

    seg_tokens = pref_tok(seg).custom["tokens"].tolist()
    seg_prefix = pref_tok(seg).custom["prefix"].tolist()

    assert seg_text == expected_output

    assert seg_tokens == tokens, f"Expected tokens {tokens}, got {seg_tokens}"
    assert seg_prefix == prefix, f"Expected prefix {prefix}, got {seg_prefix}"
