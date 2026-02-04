import pytest

from caiman_asr_train.train_utils.lr import lr_policy


class DummyOptimizer:
    def __init__(self, param_groups):
        self.param_groups = param_groups


@pytest.mark.parametrize(
    "step,warmup,expected_lr",
    [
        (0, 3, 0.1),
        (1, 3, 0.2),
    ],
)
def test_lr_policy_warmup_increases(step, warmup, expected_lr):
    optimizer = DummyOptimizer([{"lr": 0.0}])

    lr_policy(
        optimizer,
        initial_lr=0.4,
        min_lr=0.0,
        step=step,
        warmup_steps=warmup,
        hold_steps=1,
        half_life_steps=10,
    )

    assert optimizer.param_groups[0]["lr"] == expected_lr


def test_lr_policy_hold_stage_multiple_groups():
    optimizer = DummyOptimizer([{"lr": 0.0}, {"lr": 0.0}])

    lr_policy(
        optimizer,
        initial_lr=[0.2, 0.4],
        min_lr=0.0,
        step=3,
        warmup_steps=3,
        hold_steps=2,
        half_life_steps=6,
    )

    assert optimizer.param_groups[0]["lr"] == 0.2
    assert optimizer.param_groups[1]["lr"] == 0.4


def test_lr_policy_decay_clamped_to_minimum():
    optimizer = DummyOptimizer([{"lr": 0.0}])

    lr_policy(
        optimizer,
        initial_lr=0.4,
        min_lr=0.12,
        step=200,
        warmup_steps=2,
        hold_steps=1,
        half_life_steps=2,
    )

    assert optimizer.param_groups[0]["lr"] == 0.12


def test_lr_policy_raises_when_initial_lr_mismatch():
    optimizer = DummyOptimizer([{"lr": 0.0}, {"lr": 0.0}])

    with pytest.raises(AssertionError):
        lr_policy(
            optimizer,
            initial_lr=0.3,
            min_lr=0.0,
            step=0,
            warmup_steps=1,
            hold_steps=1,
            half_life_steps=1,
        )
