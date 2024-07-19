import pytest

from caiman_asr_train.rnnt.config import grad_noise_scheduler, tokenizer


@pytest.mark.parametrize(
    "ex_input, expected",
    [
        (
            {"tokenizer": {"labels": "labs", "sentpiece_model": "foo"}},
            {"labels": "labs", "sentpiece_model": "foo", "sampling": 0.0},
        )
    ],
)
def test_tokenizer(ex_input, expected):
    """ """
    returned = tokenizer(ex_input)
    assert returned == expected


@pytest.mark.parametrize(
    "ex_input, raise_error",
    [
        (
            {"tokenizer": {"labels": "labs", "sentpiece_model": None, "sampling": 2.0}},
            AssertionError,
        ),
        (
            {
                "tokenizer": {
                    "labels": "labs",
                    "sentpiece_model": None,
                    "sampling": "three",
                }
            },
            ValueError,
        ),
        ({"tokenizer": {}}, AssertionError),
    ],
)
def test_tokenizer_raises(ex_input, raise_error):
    with pytest.raises(raise_error):
        tokenizer(ex_input)


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
