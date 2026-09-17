import pytest
import torch

from caiman_asr_train.train_utils.grad_noise_scheduler import (
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


class EncoderOnlyModel(torch.nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(dim, dim), torch.nn.Linear(dim, dim)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_add_grad_noise_perturbs_every_encoder_grad():
    """The old implementation shadowed the float std with the sampled tensor,
    so the second parameter's torch.normal call got a mismatched-shape tensor
    std and raised."""
    model = EncoderOnlyModel().cuda()
    for p in model.parameters():
        p.grad = torch.zeros_like(p)

    scheduler = GradNoiseScheduler(seed=1, noise_level=0.1, start_step=1)
    _, noise_level = scheduler.add_grad_noise(model, step=5, world_size=1)

    assert noise_level == scheduler.std(5)
    for p in model.encoder.parameters():
        assert p.grad.abs().sum() > 0, "every encoder grad must be perturbed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_add_grad_noise_respects_world_size_and_amp_scale():
    """Noise added to SCALED gradients must be multiplied by the loss scale
    (effective unscaled std == scheduler.std), and the world_size division
    must actually apply (it used to discard its result)."""
    world_size, scale = 2, 8.0
    model = EncoderOnlyModel().cuda()
    for p in model.parameters():
        p.grad = torch.zeros_like(p)

    scheduler = GradNoiseScheduler(seed=1, noise_level=0.1, start_step=1)
    scheduler.add_grad_noise(model, step=1, world_size=world_size, scale=scale)

    expected_std = scheduler.std(1) * scale / world_size
    samples = torch.cat([p.grad.flatten() for p in model.encoder.parameters()])
    measured = samples.std().item()
    assert measured == pytest.approx(expected_std, rel=0.1), (
        f"grad noise std {measured} != expected {expected_std} "
        f"(std={scheduler.std(1)}, scale={scale}, world_size={world_size})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_add_grad_noise_off_before_start_step():
    model = EncoderOnlyModel().cuda()
    for p in model.parameters():
        p.grad = torch.zeros_like(p)

    scheduler = GradNoiseScheduler(seed=1, noise_level=0.1, start_step=10)
    _, noise_level = scheduler.add_grad_noise(model, step=5, world_size=1)

    assert noise_level == 0
    for p in model.encoder.parameters():
        assert p.grad.abs().sum() == 0
