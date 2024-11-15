import pytest
from pydantic import ValidationError

from caiman_asr_train.data.unk_handling import UnkHandling
from caiman_asr_train.rnnt.config import get_tokenizer_conf, grad_noise_scheduler


@pytest.mark.parametrize(
    "ex_input, expected",
    [
        (
            {"tokenizer": {"labels": list("labs"), "sentpiece_model": "foo"}},
            {
                "labels": list("labs"),
                "sentpiece_model": "foo",
                "sampling": 0.0,
                "unk_handling": UnkHandling.FAIL,
            },
        )
    ],
)
def test_tokenizer(ex_input, expected):
    """ """
    returned = get_tokenizer_conf(ex_input)
    assert returned == expected


@pytest.mark.parametrize(
    "ex_input, raise_error,message",
    [
        (
            {
                "tokenizer": {
                    "labels": "labs",
                    "sentpiece_model": "fake.model",
                    "sampling": 0.5,
                }
            },
            ValidationError,
            "1 validation error for TokenizerConfig\nlabels\n  "
            "Input should be a valid list",
        ),
        (
            {
                "tokenizer": {
                    "labels": ["l", "a", "b", "s"],
                    "sentpiece_model": None,
                    "sampling": 0.5,
                }
            },
            ValidationError,
            "1 validation error for TokenizerConfig\nsentpiece_model\n  "
            "Input should be a valid string",
        ),
        (
            {
                "tokenizer": {
                    "labels": ["l", "a", "b", "s"],
                    "sentpiece_model": "fake.model",
                    "sampling": "three",
                }
            },
            ValidationError,
            "1 validation error for TokenizerConfig\nsampling\n  "
            "Input should be a valid number",
        ),
        (
            {
                "tokenizer": {
                    "labels": ["l", "a", "b", "s"],
                    "sentpiece_model": "fake.model",
                    "sampling": 2.0,
                }
            },
            ValidationError,
            "1 validation error for TokenizerConfig\nsampling\n  "
            "Input should be less than or equal to 1",
        ),
        (
            {
                "tokenizer": {
                    "labels": "labs",
                    "sentpiece_model": None,
                    "sampling": 2.0,
                }
            },
            ValidationError,
            "3 validation errors for TokenizerConfig\nsentpiece_model\n  "
            "Input should be a valid string",
        ),
        (
            {
                "tokenizer": {
                    "labels": "labs",
                    "sentpiece_model": None,
                    "sampling": "three",
                }
            },
            ValidationError,
            "3 validation errors for TokenizerConfig\nsentpiece_model\n  "
            "Input should be a valid string",
        ),
        (
            {"tokenizer": {}},
            ValidationError,
            "2 validation errors for TokenizerConfig\nsentpiece_model\n  "
            "Field required",
        ),
    ],
)
def test_tokenizer_raises(ex_input, raise_error, message):
    with pytest.raises(raise_error, match=message):
        get_tokenizer_conf(ex_input)


@pytest.mark.parametrize(
    "ex_input, expected",
    [
        (
            {
                "grad_noise_scheduler": {
                    "noise_level": 0.01,
                    "decay_const": 0.01,
                    "start_step": 0,
                }
            },
            {"noise_level": 0.01, "decay_const": 0.01, "start_step": 0},
        ),
        #
        (
            {
                "grad_noise_scheduler": {
                    "noise_level": 0.0,
                    "decay_const": 0.01,
                    "start_step": 0,
                }
            },
            {"noise_level": 0.00, "decay_const": 0.01, "start_step": 0},
        ),
        #
        (
            {"grad_noise_scheduler": {}},
            {"noise_level": 0.15, "decay_const": 0.55, "start_step": 1},
        ),
        # wrong start_step value, but the validate_and_fill will not catch it.
        (
            {"grad_noise_scheduler": {"start_step": -0.50}},
            {"noise_level": 0.15, "decay_const": 0.55, "start_step": -0.5},
        ),
    ],
)
def test_grad_noise_scheduler_config(ex_input, expected):
    """ """
    returned = grad_noise_scheduler(ex_input)
    assert returned.get("noise_level") == expected.get("noise_level")
    assert returned.get("decay_const") == expected.get("decay_const")
    assert returned.get("start_step") == expected.get("start_step")
