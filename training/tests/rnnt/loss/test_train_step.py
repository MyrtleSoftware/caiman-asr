"""The fused ``--transducer_backend tilelang`` training step against Apex+triton.

``test_equivalence`` covers the kernels; what is new here is the seam: ``weight.t()``
routing ``dW`` back to the fp32 parameter, the in-graph cast of ``detach_to_bf``'s fp32
leaves, and the ``FusedJointLoss`` view's train/eval mode.
"""

import copy
from argparse import Namespace

import pytest
import torch
from tests.sm80 import skip_without_tilelang

skip_without_tilelang()

from tests.tile_transducer.kernels import untuned_kernels  # noqa: E402

from caiman_asr_train.core.batches import BatchFirst, TimeFirst  # noqa: E402
from caiman_asr_train.rnnt.joint_loss import FusedJointLoss  # noqa: E402
from caiman_asr_train.rnnt.loss import (  # noqa: E402
    IDENTITY_LOSS_MODIFIERS,
    ApexTransducerLoss,
    FusedProjAndLoss,
    LossModifiers,
)
from caiman_asr_train.rnnt.sub_models import RNNTSubModels  # noqa: E402
from caiman_asr_train.tile_transducer.autotune import Mode  # noqa: E402
from caiman_asr_train.train_utils.batch_splitting import (  # noqa: E402
    detach_to_bf,
    train_step,
)
from caiman_asr_train.utils.seed import set_seed  # noqa: E402

RTOL = ATOL = 2e-2
JOINT_DTYPE = torch.float32


@pytest.fixture()
def n_classes_fixture():
    """The kernels need V % block_V == 0; the conftest default of 100 does not."""
    return 128


@pytest.fixture()
def joint_n_hid_fixture():
    """The fused forward needs H % block_H == 0; the conftest default of 13 does not."""
    return 64


@pytest.fixture()
def args():
    return Namespace(
        no_amp=True,
        grad_accumulation_batches=1,
        joint_lattice_cap=2**17,
        batch_split_factor=1,
        num_gpus=1,
        conditional=False,
    )


@pytest.fixture()
def make_fused_loss(n_classes_fixture, joint_n_hid_fixture):
    def build(dtype=JOINT_DTYPE, eos_idx=None):
        blank = n_classes_fixture - 1
        return FusedProjAndLoss(
            kernels=untuned_kernels(
                Mode.fused, joint_n_hid_fixture, n_classes_fixture, blank, dtype
            ),
            blank_idx=blank,
            eos_idx=eos_idx,
        )

    return build


@pytest.fixture()
def make_triton_loss(n_classes_fixture):
    def build(eos_idx=None):
        return ApexTransducerLoss(
            backend="triton",
            blank_idx=n_classes_fixture - 1,
            eos_idx=eos_idx,
            star_idx=None,
            packed_input=True,
            validate_first_n_remaining=0,
        )

    return build


@pytest.fixture()
@torch.no_grad()
def input_data(input_dim_fixture, n_classes_fixture):
    """Ragged batch; audio lengths are multiples of enc_stack_time_factor (2)."""
    set_seed(seed=0, local_rank=0)
    dev = "cuda"
    feats = torch.randn(24, 5, input_dim_fixture, device=dev)
    feat_lens = torch.tensor([24, 20, 24, 12, 8], device=dev, dtype=torch.int)
    txt = torch.randint(0, n_classes_fixture - 1, (5, 10), device=dev)
    txt_lens = torch.tensor([10, 4, 7, 10, 1], device=dev, dtype=torch.int)
    return TimeFirst(feats, feat_lens), BatchFirst(txt, txt_lens).T


def encoder_outputs(model, feats, tokens):
    """Fresh fp32 leaves: the exact inputs the joint+loss sees in a train step."""
    f, _ = model.encode(state=None, prefix=None, feats=feats)
    g, _ = model.predict(state=model.sos_for(feats), prefix=None, tokens=tokens)
    return detach_to_bf(f), detach_to_bf(g)


def joint_loss_view(model, loss_fn, dropout_prob=0.0):
    """A throwaway view slaved to the model's mode, as ``RNNTSubModels`` does."""
    view = FusedJointLoss(model, loss_fn, dropout_prob=dropout_prob)
    view.train(model.training)
    return view


def step(model, args, feats, tokens, mods=IDENTITY_LOSS_MODIFIERS, scaler=None):
    return train_step(
        model,
        args,
        feats,
        tokens,
        scaler=scaler,
        rnnt_state=None,
        loss_mods=mods,
        prefix=None,
        is_last_microbatch=True,
    )


def assert_joint_grads_match(fused, ref):
    ref_grads = dict(ref.named_parameters())
    joint_ids = {id(p) for p in fused.joint_loss.parameters()}
    checked = {"joint": 0, "bridge": 0}
    for name, param in fused.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has a non-finite gradient"
        assert param.grad.abs().sum().item() > 0.0, f"{name} gradient is all zero"
        if id(param) in joint_ids:
            torch.testing.assert_close(
                param.grad, ref_grads[name].grad, rtol=RTOL, atol=ATOL
            )
            checked["joint"] += 1
        else:
            checked["bridge"] += 1
    assert checked["joint"] > 0 and checked["bridge"] > 0, checked


def test_fused_matches_apex_triton_joint_loss(
    model_factory, make_fused_loss, make_triton_loss, input_data
):
    """Same detached f/g -> same loss and same grads on f, g and the projection."""
    feats, tokens = input_data
    model_fused = model_factory().cuda()
    model_ref = copy.deepcopy(model_fused)
    t_b = tokens.T.contiguous()

    f_a, g_a = encoder_outputs(model_fused, feats, tokens)
    loss_fused = joint_loss_view(model_fused, make_fused_loss())(
        f_a, g_a, t_b, IDENTITY_LOSS_MODIFIERS
    )
    loss_fused.backward()

    f_b, g_b = encoder_outputs(model_ref, feats, tokens)
    logits, batch_offset = model_ref.joint(f_b, g_b)
    loss_ref = make_triton_loss()(
        logits,
        f_b.lens,
        t_b.data,
        t_b.lens,
        batch_offset=batch_offset,
        max_f_len=f_b.lens.max().item(),
        loss_mods=IDENTITY_LOSS_MODIFIERS,
    )
    loss_ref.backward()

    torch.testing.assert_close(
        loss_fused.detach(), loss_ref.detach(), rtol=RTOL, atol=ATOL
    )

    t = torch.arange(f_a.data.shape[1], device="cuda")[None, :]
    u = torch.arange(g_a.data.shape[1], device="cuda")[None, :]
    f_mask = (t < feats.lens[:, None]).unsqueeze(-1)
    g_mask = (u < (t_b.lens + 1)[:, None]).unsqueeze(-1)
    torch.testing.assert_close(
        f_a.data.grad * f_mask, f_b.data.grad * f_mask, rtol=RTOL, atol=ATOL
    )
    torch.testing.assert_close(
        g_a.data.grad * g_mask, g_b.data.grad * g_mask, rtol=RTOL, atol=ATOL
    )

    lin_a, lin_b = model_fused.joint_net[-1], model_ref.joint_net[-1]
    assert lin_a.weight.grad is not None, "dW never reached the fp32 parameter"
    assert lin_a.weight.grad.dtype is torch.float32
    torch.testing.assert_close(
        lin_a.weight.grad, lin_b.weight.grad, rtol=RTOL, atol=ATOL
    )
    torch.testing.assert_close(lin_a.bias.grad, lin_b.bias.grad, rtol=RTOL, atol=ATOL)


def test_backward_scale_scales_the_gradients(
    model_factory, make_fused_loss, input_data
):
    """An upstream ``.backward(scale * ...)`` must be linear in the gradients."""
    feats, tokens = input_data
    t_b = tokens.T.contiguous()
    model = model_factory().cuda()
    loss_fn = make_fused_loss()

    grads = {}
    for scale in (1.0, 4.0):
        f, g = encoder_outputs(model, feats, tokens)
        model.joint_net[-1].zero_grad(set_to_none=True)
        loss = joint_loss_view(model, loss_fn)(f, g, t_b, IDENTITY_LOSS_MODIFIERS)
        loss.backward(torch.full_like(loss, scale))
        grads[scale] = (
            f.data.grad.clone(),
            model.joint_net[-1].weight.grad.clone(),
            loss.detach().clone(),
        )

    df_1, dW_1, loss_1 = grads[1.0]
    df_4, dW_4, loss_4 = grads[4.0]
    torch.testing.assert_close(df_4, df_1 * 4, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(dW_4, dW_1 * 4, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(loss_4, loss_1, rtol=0, atol=0)


def test_dropout_follows_the_model_mode(model_factory, make_fused_loss, input_data):
    """The view owns an Apex joint that is not a registered submodule of the RNNT, so
    callers must slave its mode to the model's or validation applies dropout."""
    feats, tokens = input_data
    t_b = tokens.T.contiguous()
    model = model_factory().cuda()
    loss_fn = make_fused_loss()

    with torch.no_grad():
        f, g = encoder_outputs(model, feats, tokens)

        model.eval()
        fused = joint_loss_view(model, loss_fn, dropout_prob=0.5)
        loss_eval = fused(f, g, t_b, IDENTITY_LOSS_MODIFIERS)
        assert not fused.training, "view mode did not follow model.eval()"
        loss_ref = joint_loss_view(model, loss_fn)(f, g, t_b, IDENTITY_LOSS_MODIFIERS)
        torch.testing.assert_close(loss_eval, loss_ref, rtol=1e-2, atol=1e-2)

        model.train()
        fused = joint_loss_view(model, loss_fn, dropout_prob=0.5)
        loss_train = fused(f, g, t_b, IDENTITY_LOSS_MODIFIERS)
        assert fused.training, "view mode did not follow model.train()"
        assert not torch.equal(loss_train, loss_ref), "dropout had no effect in train"


@pytest.mark.parametrize("batch_split_factor", [1, 3])
def test_fused_strategy_matches_apex_strategy(
    model_factory,
    make_fused_loss,
    make_triton_loss,
    input_data,
    args,
    batch_split_factor,
):
    """End to end ``train_step``; bsf=3 does not divide the batch of 5, covering
    ``_get_bsplit``'s rounded-up final split."""
    feats, tokens = input_data
    base = model_factory().cuda()
    model_fused = RNNTSubModels.from_RNNT(
        copy.deepcopy(base), ddp=False, loss_fn=make_fused_loss()
    )
    model_ref = RNNTSubModels.from_RNNT(base, ddp=False, loss_fn=make_triton_loss())
    args.batch_split_factor = batch_split_factor

    loss_fused, nan_fused, state_fused = step(model_fused, args, feats, tokens)
    loss_ref, nan_ref, state_ref = step(model_ref, args, feats, tokens)

    assert not nan_fused and not nan_ref
    assert (state_fused is None) == (state_ref is None)
    assert loss_fused == pytest.approx(loss_ref, rel=1e-2, abs=1e-2)
    assert_joint_grads_match(model_fused, model_ref)


def test_train_step_applies_the_penalty(
    model_factory,
    make_fused_loss,
    make_triton_loss,
    input_data,
    args,
    n_classes_fixture,
):
    """The penalty must reach the fused step, agree with Apex+triton, and move the
    loss away from the unpenalised one."""
    feats, tokens = input_data
    eos_idx = n_classes_fixture - 2
    for b in range(tokens.lens.shape[0]):
        last = int(tokens.lens[b].item()) - 1
        if last >= 0:
            tokens.data[last, b] = eos_idx

    base = model_factory().cuda()
    fused_loss = make_fused_loss(eos_idx=eos_idx)
    mods = LossModifiers(dp_lam=0.05, eos_lam=0.1, star_penalty=1.0)

    model_fused = RNNTSubModels.from_RNNT(
        copy.deepcopy(base), ddp=False, loss_fn=fused_loss
    )
    model_ref = RNNTSubModels.from_RNNT(
        copy.deepcopy(base), ddp=False, loss_fn=make_triton_loss(eos_idx=eos_idx)
    )
    model_identity = RNNTSubModels.from_RNNT(base, ddp=False, loss_fn=fused_loss)

    loss_fused, nan_fused, _ = step(model_fused, args, feats, tokens, mods)
    loss_ref, nan_ref, _ = step(model_ref, args, feats, tokens, mods)
    loss_identity, nan_identity, _ = step(model_identity, args, feats, tokens)

    assert not (nan_fused or nan_ref or nan_identity)
    assert loss_fused == pytest.approx(loss_ref, rel=1e-2, abs=1e-2)
    assert_joint_grads_match(model_fused, model_ref)
    assert abs(loss_fused - loss_identity) > 1e-2


@pytest.mark.parametrize("no_amp", [True, False])
def test_amp_state_picks_the_joint_precision(
    model_factory, make_fused_loss, input_data, args, no_amp
):
    """``f``/``g`` arrive fp32 either way, so the view must read the AMP state itself;
    a wrong read trips the kernels' dtype assert."""
    feats, tokens = input_data
    expected = torch.float32 if no_amp else torch.float16

    def run(dtype):
        model = RNNTSubModels.from_RNNT(
            model_factory().cuda(), ddp=False, loss_fn=make_fused_loss(dtype=dtype)
        )
        return step(
            model,
            Namespace(**vars(args) | dict(no_amp=no_amp)),
            feats,
            tokens,
            scaler=None if no_amp else torch.amp.GradScaler("cuda"),
        )

    _, nan, _ = run(expected)
    assert not nan
    with pytest.raises(AssertionError, match="tuned for"):
        run(torch.float16 if no_amp else torch.float32)


def test_star_penalty_value_is_ignored(
    model_factory, make_fused_loss, input_data, args
):
    """The fused path has no star channel, so ``star_penalty`` must be a no-op, not a
    rejection: the scheduler computes a value every step regardless of the vocab."""
    feats, tokens = input_data
    models = [
        RNNTSubModels.from_RNNT(
            model_factory().cuda(), ddp=False, loss_fn=make_fused_loss()
        )
        for _ in range(2)
    ]
    starred = LossModifiers(dp_lam=0.0, eos_lam=0.0, star_penalty=0.5)

    loss_a, nan_a, _ = step(models[0], args, feats, tokens, starred)
    loss_b, nan_b, _ = step(models[1], args, feats, tokens)
    assert not (nan_a or nan_b)
    assert loss_a == pytest.approx(loss_b, rel=1e-2, abs=1e-2)
