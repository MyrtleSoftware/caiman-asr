#!/usr/bin/env python3
from argparse import Namespace

import pytest
import torch
from torch.amp import GradScaler

from caiman_asr_train.train_utils.optimizer import OptimizerWrapper

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)


def make_wrapper(
    lower_bound=None,
    init_scale=2.0**16,
    no_amp=False,
    norm_skip_factor=None,
    weight_decay=0.0,
    skip_decay_momentum=0.0,
):
    model = torch.nn.Linear(4, 2).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=weight_decay)
    scaler = None if no_amp else GradScaler("cuda", init_scale=init_scale)
    if scaler is not None:
        # The scaler initializes its scale tensor lazily on the first scale()
        # call (normally scaler.scale(loss).backward() in the train step)
        scaler.scale(torch.zeros(1, device="cuda"))
    args = Namespace(no_amp=no_amp)
    wrapper = OptimizerWrapper(
        args,
        optimizer,
        scaler,
        lower_bound=lower_bound,
        norm_skip_factor=norm_skip_factor,
        skip_decay_momentum=skip_decay_momentum,
    )
    return model, scaler, wrapper


def set_grads(model, value: float) -> None:
    for p in model.parameters():
        p.grad = torch.full_like(p, value)


def weights_of(model) -> torch.Tensor:
    return torch.cat([p.detach().flatten().clone() for p in model.parameters()])


def test_finite_grads_step_and_keep_scale():
    model, scaler, wrapper = make_wrapper(init_scale=8.0)
    before = weights_of(model)
    set_grads(model, 1.0)
    stepped = wrapper.step(total_norm=1.0)
    assert stepped is True
    assert not torch.equal(weights_of(model), before), "optimizer should step"
    assert scaler.get_scale() == 8.0


def test_inf_grads_skip_and_backoff_scale():
    """The stepped flag gates the EMA blend in the train loop, so it must
    report skipped steps faithfully."""
    model, scaler, wrapper = make_wrapper(init_scale=8.0)
    before = weights_of(model)
    set_grads(model, float("inf"))
    stepped = wrapper.step(total_norm=float("inf"))
    assert stepped is False
    assert torch.equal(weights_of(model), before), "step must be skipped"
    assert scaler.get_scale() == 4.0, "scale should back off once"


def test_nan_grads_report_skipped():
    model, _, wrapper = make_wrapper(init_scale=8.0)
    before = weights_of(model)
    set_grads(model, float("nan"))
    assert wrapper.step(total_norm=float("nan")) is False
    assert torch.equal(weights_of(model), before)


def test_lower_bound_pins_scale_immediately():
    """The scale must never be left below the bound at the end of step();
    the old implementation deferred the pin to the next step, so one whole
    optimizer step ran with an out-of-bounds scale."""
    model, scaler, wrapper = make_wrapper(lower_bound=8.0, init_scale=8.0)
    set_grads(model, float("inf"))
    wrapper.step(total_norm=float("inf"))  # backoff would give 4.0
    assert scaler.get_scale() == 8.0, "scale must be pinned in the same step"


def test_lower_bound_inactive_above_bound():
    model, scaler, wrapper = make_wrapper(lower_bound=2.0, init_scale=8.0)
    set_grads(model, float("inf"))
    wrapper.step(total_norm=float("inf"))
    assert scaler.get_scale() == 4.0, "backoff above the bound is untouched"


def test_norm_skip_factor_drops_huge_finite_norms():
    """Finite-but-huge gradients pass the GradScaler's inf check, so LAMB's
    trust ratio would take a full-sized step in their direction; the relative
    guard must drop the update without touching the scale."""
    model, scaler, wrapper = make_wrapper(init_scale=8.0, norm_skip_factor=10.0)
    before = weights_of(model)
    set_grads(model, 1.0)

    # Establish the running-average norm with a few normal steps
    for _ in range(3):
        assert wrapper.step(total_norm=5.0) is True

    before = weights_of(model)
    assert wrapper.step(total_norm=5000.0) is False, "1000x the EMA must skip"
    assert torch.equal(weights_of(model), before), "huge norm must skip"
    assert scaler.get_scale() == 8.0, "no overflow: scale must be untouched"

    assert wrapper.step(total_norm=6.0) is True, "normal norm must step again"
    assert not torch.equal(weights_of(model), before)


def test_norm_skip_factor_tracks_drifting_norms():
    """Healthy norms span orders of magnitude over a run (O(8000) in warmup,
    O(10) converged); the guard is relative to a running average precisely so
    large-but-typical early norms are NOT skipped."""
    model, _, wrapper = make_wrapper(init_scale=8.0, norm_skip_factor=10.0)
    set_grads(model, 1.0)

    # Warmup-sized norms: the first observation seeds the average, so an
    # 8000-norm step early in training steps normally.
    assert wrapper.step(total_norm=8000.0) is True
    assert wrapper.step(total_norm=6000.0) is True

    # Rejected garbage must not drag the average up: after a skip, the
    # average still reflects the accepted steps only.
    assert wrapper.step(total_norm=100_000.0) is False
    assert wrapper.step(total_norm=7000.0) is True


def test_norm_skip_factor_lets_scaler_handle_non_finite():
    """Non-finite norms must fall through to the GradScaler so the scale still
    backs off (a relative skip would leave the scale stuck too high)."""
    model, scaler, wrapper = make_wrapper(init_scale=8.0, norm_skip_factor=10.0)
    set_grads(model, 1.0)
    assert wrapper.step(total_norm=5.0) is True  # seed the running average

    before = weights_of(model)
    set_grads(model, float("inf"))
    assert wrapper.step(total_norm=float("inf")) is False
    assert torch.equal(weights_of(model), before)
    assert scaler.get_scale() == 4.0, "inf grads must still back the scale off"


def test_no_amp_skips_on_non_finite_norm():
    model, _, wrapper = make_wrapper(no_amp=True)
    before = weights_of(model)
    set_grads(model, 1.0)

    assert wrapper.step(total_norm=float("nan")) is False
    assert torch.equal(weights_of(model), before), "non-finite norm must skip"

    assert wrapper.step(total_norm=1.0) is True
    assert not torch.equal(weights_of(model), before), "finite norm must step"


def test_skipped_step_applies_weight_decay():
    """A skipped step must still apply decoupled weight decay: during a
    sustained instability the skips otherwise freeze the runaway weight
    norms exactly where they exploded, making the stall self-sustaining."""
    model, scaler, wrapper = make_wrapper(init_scale=8.0, weight_decay=0.01)
    before = weights_of(model)
    set_grads(model, float("inf"))

    assert wrapper.step(total_norm=float("inf")) is False
    expected = before * (1 - 0.1 * 0.01)  # lr * weight_decay
    assert torch.allclose(weights_of(model), expected), "decay must apply on skip"
    assert scaler.get_scale() == 4.0, "scaler backoff is unchanged"


def test_norm_skip_applies_weight_decay_without_scale_change():
    model, scaler, wrapper = make_wrapper(
        init_scale=8.0, norm_skip_factor=10.0, weight_decay=0.01
    )
    set_grads(model, 1.0)
    assert wrapper.step(total_norm=5.0) is True  # seed the norm EMA

    before = weights_of(model)
    assert wrapper.step(total_norm=5000.0) is False
    expected = before * (1 - 0.1 * 0.01)
    assert torch.allclose(weights_of(model), expected)
    assert scaler.get_scale() == 8.0, "no overflow: scale must be untouched"


def test_no_amp_skip_applies_weight_decay():
    model, _, wrapper = make_wrapper(no_amp=True, weight_decay=0.01)
    before = weights_of(model)
    set_grads(model, 1.0)

    assert wrapper.step(total_norm=float("nan")) is False
    expected = before * (1 - 0.1 * 0.01)
    assert torch.allclose(weights_of(model), expected)


def test_skip_decay_no_op_for_zero_weight_decay():
    """Groups the optimizer would not decay must be untouched on skips (the
    default-wd-0 skip tests above also rely on this bit-exactly)."""
    model, _, wrapper = make_wrapper(init_scale=8.0, weight_decay=0.0)
    before = weights_of(model)
    set_grads(model, float("inf"))

    assert wrapper.step(total_norm=float("inf")) is False
    assert torch.equal(weights_of(model), before)


def test_norm_skip_factor_zero_disables_the_check():
    """0 (the CLI's disable value) must behave like the guard being absent."""
    model, _, wrapper = make_wrapper(init_scale=8.0, norm_skip_factor=0.0)
    before = weights_of(model)
    set_grads(model, 1.0)
    assert wrapper.step(total_norm=5.0) is True  # seeds the EMA
    assert wrapper.step(total_norm=1e9) is True, "disabled guard must not skip"
    assert not torch.equal(weights_of(model), before)


def test_guard_arg_defaults_are_floats():
    """argparse applies `type=` only to CLI strings, never to defaults: an
    int default flows through setup (2**6 stays int) into the beartype-
    checked Optional[float] wrapper params and crashes at startup. Caught
    by the mini-train CI; pinned here."""
    from caiman_asr_train.args.train import train_arg_parser

    parser = train_arg_parser()
    assert isinstance(parser.get_default("grad_scaler_lower_bound_log2"), float)
    assert isinstance(parser.get_default("grad_norm_skip_factor"), float)
    assert isinstance(parser.get_default("skip_decay_momentum"), float)


def test_skip_decay_momentum_zero_matches_fixed_decay():
    """momentum=0 keeps the velocity at 1 on every skip -- i.e. the
    un-augmented fixed lr*weight_decay applied once per skip."""
    model, _, wrapper = make_wrapper(
        init_scale=8.0, weight_decay=0.01, skip_decay_momentum=0.0
    )
    before = weights_of(model)
    set_grads(model, float("inf"))
    for _ in range(3):
        assert wrapper.step(total_norm=float("inf")) is False
    expected = before * (1 - 0.1 * 0.01) ** 3
    assert torch.allclose(weights_of(model), expected)


def test_skip_decay_momentum_accelerates_over_consecutive_skips():
    """momentum=1 ramps the velocity linearly (1, 2, 3, ...), so consecutive
    skips decay progressively harder than the fixed rate."""
    model, _, wrapper = make_wrapper(
        init_scale=8.0, weight_decay=0.01, skip_decay_momentum=1.0
    )
    before = weights_of(model)
    set_grads(model, float("inf"))
    for _ in range(3):
        assert wrapper.step(total_norm=float("inf")) is False
    lrwd = 0.1 * 0.01
    expected = before * (1 - lrwd * 1) * (1 - lrwd * 2) * (1 - lrwd * 3)
    assert torch.allclose(weights_of(model), expected)
    # strictly more decay than the fixed (momentum=0) rate over the same skips
    fixed = before * (1 - lrwd) ** 3
    assert (weights_of(model).abs() <= fixed.abs() + 1e-9).all()


def test_skip_decay_velocity_resets_on_real_step():
    """A real optimizer step ends the skip run: the velocity returns to 0, so
    the next skip decays at the base rate (velocity 1) again, not the
    accumulated rate."""
    model, _, wrapper = make_wrapper(
        init_scale=8.0, weight_decay=0.01, skip_decay_momentum=1.0
    )
    # two skips build the velocity up to 2
    set_grads(model, float("inf"))
    assert wrapper.step(total_norm=float("inf")) is False
    assert wrapper.step(total_norm=float("inf")) is False
    # a real step must reset the velocity
    set_grads(model, 1.0)
    assert wrapper.step(total_norm=1.0) is True
    # the next skip therefore decays at velocity 1 (base rate), not 3
    after_step = weights_of(model)
    set_grads(model, float("inf"))
    assert wrapper.step(total_norm=float("inf")) is False
    expected = after_step * (1 - 0.1 * 0.01 * 1)
    assert torch.allclose(weights_of(model), expected)
