import kenlm
import pytest

from caiman_asr_train.lm.kenlm_ngram import KenLmModel


@pytest.fixture()
def ngram_lm(ngram_path):
    return KenLmModel(ngram_path)


@pytest.mark.parametrize(
    "input_str, expected_score",
    [
        ("▁so", -6.120774573414487),
        ("▁the", -2.4116030028837527),
        ("▁a", -3.7637535396913835),
        ("a", -6.008470474834914),
        ("_a", -6.437767163738605),
        ("<s>", -0.2525948068386439),
        ("1111", -6.437767163738605),
        ("<unk>", -6.437767163738605),
        ("?", -6.437767163738605),
    ],
)
def test_kenlm_ngram(ngram_lm, input_str, expected_score):
    init_ngram_state = kenlm.State()
    ngram_lm.model.BeginSentenceWrite(init_ngram_state)
    score, _ = ngram_lm.score_ngram(input_str, init_ngram_state)
    assert score == expected_score
