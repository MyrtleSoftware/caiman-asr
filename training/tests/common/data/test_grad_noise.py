import pytest

from rnnt_train.common.data.grad_noise_scheduler import (
    GradNoiseScheduler,
    switch_on_grad_noise_scheduler,
)


@pytest.mark.parametrize(
    "ex_input, expected",
    [
        ({}, {"start_step": 1}),
        ({"seed": 2}, {"start_step": 1}),
    ],
)
def test_grad_noise_scheduler(ex_input, expected):
    obj = GradNoiseScheduler(**ex_input)
    assert obj.start_step == expected["start_step"]


@pytest.mark.parametrize(
    "cfg_input",
    [
        {"grad_noise_scheduler": {}},
        {"grad_noise_scheduler": {"noise_level": 0.00}},
        {"grad_noise_scheduler": {"noise_level": -1.0}},
    ],
)
@pytest.mark.parametrize("enc_freeze", [True, False])
def test_switch_on_grad_noise_sched_return_false(cfg_input, enc_freeze):
    gns = switch_on_grad_noise_scheduler(cfg_input, enc_freeze)
    assert gns is False


@pytest.mark.parametrize(
    "cfg_input",
    [
        {"grad_noise_scheduler": {"noise_level": 0.05}},
    ],
)
def test_switch_on_grad_noise_sched_return_true(cfg_input):
    gns = switch_on_grad_noise_scheduler(cfg_input, enc_freeze=False)
    assert gns is True
