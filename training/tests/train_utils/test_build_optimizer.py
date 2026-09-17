from argparse import Namespace

import pytest
import torch
from apex.optimizers import FusedLAMB

from caiman_asr_train.train_utils.build_optimizer import build_optimizer
from caiman_asr_train.train_utils.optimizer import OptimizerWrapper


def test_build_optimizer(mini_model_factory):
    model, _ = mini_model_factory()
    args = Namespace(lr=0.001, weight_decay=0.01, beta1=0.9, beta2=0.999, clip_norm=1.0)
    optimizer = build_optimizer(args, model)
    assert isinstance(optimizer, FusedLAMB)


def test_param_groups_carry_weight_decay(mini_model_factory):
    """OptimizerWrapper._decay_on_skip reads group["weight_decay"] with a
    silent 0.0 fallback: if FusedLAMB's param groups ever stopped carrying
    the key (or build_optimizer stopped propagating it), decay-on-skip
    would silently become a no-op in production while the SGD-based unit
    tests stayed green. Pin the contract here, on the real optimizer."""
    model, _ = mini_model_factory()
    args = Namespace(lr=0.001, weight_decay=0.01, beta1=0.9, beta2=0.999, clip_norm=1.0)
    optimizer = build_optimizer(args, model)

    assert len(optimizer.param_groups) > 0
    for group in optimizer.param_groups:
        assert "weight_decay" in group, "decay-on-skip depends on this key"
        assert group["weight_decay"] == args.weight_decay
        assert "lr" in group and group["lr"] > 0


def test_decay_on_skip_acts_through_fused_lamb(mini_model_factory):
    """End-to-end: a scaler-skipped step must shrink every parameter by its
    own group's (1 - lr * weight_decay) when driven by the production
    FusedLAMB optimizer, not just the SGD stand-in used elsewhere."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model, _ = mini_model_factory()
    model = model.cuda()
    args = Namespace(
        lr=0.001,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.999,
        clip_norm=1.0,
        no_amp=False,
    )
    optimizer = build_optimizer(args, model)
    scaler = torch.amp.GradScaler("cuda")
    scaler.scale(torch.zeros(1, device="cuda"))  # initialise the lazy scale
    wrapper = OptimizerWrapper(args, optimizer, scaler)

    before = {}
    for gi, group in enumerate(optimizer.param_groups):
        for pi, p in enumerate(group["params"]):
            p.grad = torch.full_like(p, float("inf"))
            before[(gi, pi)] = p.detach().clone()

    assert wrapper.step(total_norm=float("inf")) is False, "step must skip"

    for gi, group in enumerate(optimizer.param_groups):
        factor = 1.0 - group["lr"] * group["weight_decay"]
        assert factor < 1.0, "groups must carry real decay for this test"
        for pi, p in enumerate(group["params"]):
            assert torch.allclose(p.detach(), before[(gi, pi)] * factor), (
                f"param group {gi} param {pi} was not decayed on the "
                f"skipped step (expected factor {factor})"
            )
