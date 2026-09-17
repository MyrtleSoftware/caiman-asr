import copy
from argparse import Namespace

import pytest
import torch
from beartype.typing import Union
from tests.rnnt.reference_train_step import reference_train_step
from torch.nn.parallel import DistributedDataParallel as DDP

from caiman_asr_train.core.batches import BatchFirst, TimeFirst
from caiman_asr_train.core.lstm import Hidden
from caiman_asr_train.core.model import StateEncoder, StatePred
from caiman_asr_train.rnnt.loss import IDENTITY_LOSS_MODIFIERS, ApexTransducerLoss
from caiman_asr_train.rnnt.model import RNNT
from caiman_asr_train.rnnt.model_forward import val_loss
from caiman_asr_train.rnnt.sub_models import RNNTSubModels
from caiman_asr_train.train_utils.batch_splitting import get_bsplit, train_step
from caiman_asr_train.train_utils.distributed import unwrap_ddp
from caiman_asr_train.utils.seed import set_seed


def _view_param_ids(view: Union[DDP, object]) -> set:
    """ids of the parameters owned by one RNNTSubModels view (DDP-wrapped or bare).

    The view's parameter tensors are the SAME objects as the RNNT's (the views only
    re-register them), so these ids match those from ``model.named_parameters()``.
    """
    module = view.module if isinstance(view, DDP) else view
    return {id(p) for p in module.parameters()}


def _run_step(train_step_fn, model, loss_fn, args, *data, **kwargs):
    if train_step_fn is reference_train_step:
        return train_step_fn(model, loss_fn, args, *data, **kwargs)

    return train_step_fn(model, args, *data, **kwargs)


def _cat_time_first(parts: list) -> TimeFirst:
    """Concatenate ``TimeFirst`` batches along the batch dim.

    The test data is unpadded and every rank uses the same sequence lengths, so the time
    dimension already matches and a plain ``cat`` over the batch axis is exact.
    """
    return TimeFirst(
        data=torch.cat([p.data for p in parts], dim=1),  # (T, B, *) -> cat on B
        lens=torch.cat([p.lens for p in parts], dim=0),
    )


# Indirectly parametrisable forms of the conftest fixtures (defaults unchanged): the
# fused kernels need V % block_V == 0 and H % block_H == 0, which the conftest 100/13 do
# not satisfy, so the fused cases below ask for 128/64 instead.
@pytest.fixture()
def n_classes_fixture(request):
    return getattr(request, "param", 100)


@pytest.fixture()
def joint_n_hid_fixture(request):
    return getattr(request, "param", 13)


def build_unfused_loss(n_classes: int) -> ApexTransducerLoss:
    return ApexTransducerLoss(
        backend="triton",
        blank_idx=n_classes - 1,
        eos_idx=None,
        star_idx=None,
        packed_input=True,
        validate_first_n_remaining=0,
    )


AUDIO_SEQ_LEN, TXT_SEQ_LEN, BATCH_SIZE = 20, 10, 8
ENC_STACK_TIME_FACTOR = 2
CELLS_PER_UTT = (AUDIO_SEQ_LEN // ENC_STACK_TIME_FACTOR) * (TXT_SEQ_LEN + 1)
NO_SPLIT = 10**9


def build_fused_loss(n_classes: int, joint_n_hid: int):
    from tests.tile_transducer.kernels import untuned_kernels

    from caiman_asr_train.rnnt.loss import FusedProjAndLoss
    from caiman_asr_train.tile_transducer.autotune import Mode

    return FusedProjAndLoss(
        kernels=untuned_kernels(
            Mode.fused, joint_n_hid, n_classes, n_classes - 1, torch.float32
        ),
        blank_idx=n_classes - 1,
        eos_idx=None,
    )


@pytest.fixture()
def loss_fn(n_classes_fixture):
    return build_unfused_loss(n_classes_fixture)


@pytest.fixture()
def args():
    return Namespace(
        grad_accumulation_batches=1,
        joint_lattice_cap=CELLS_PER_UTT,
        num_gpus=1,
        delay_penalty=0.0,
        conditional=False,
    )


@pytest.fixture()
def device():
    return "cuda"


@pytest.fixture()
@torch.no_grad()
def input_data_factory(input_dim_fixture, n_classes_fixture, device):
    def _gen_data(seed=0, local_rank=0):
        """Unpadded, uniform lengths."""
        set_seed(seed=seed, local_rank=local_rank)

        feats = torch.randn(AUDIO_SEQ_LEN, BATCH_SIZE, input_dim_fixture, device=device)
        feat_lens = torch.tensor(
            [AUDIO_SEQ_LEN] * BATCH_SIZE, device=device, dtype=torch.int
        )
        txt = torch.randint(
            0, n_classes_fixture - 1, (BATCH_SIZE, TXT_SEQ_LEN), device=device
        )
        txt_lens = torch.tensor(
            [TXT_SEQ_LEN] * BATCH_SIZE, device=device, dtype=torch.int
        )

        return TimeFirst(feats, feat_lens), BatchFirst(txt, txt_lens).T

    return _gen_data


def build_objects(
    model_factory,
    input_data_factory,
    args,
    device,
    no_amp,
    loss_fn,
    local_rank=0,
):
    """
    Create identical pair of (models, scaler, data) objects for batch splitting on & off.

    Note that the models are not identical as the batch splitting code requires the
    RNNTSubModels class to be used but the weights are the same and the forward
    (and backwards) passes should be equivalent.
    """
    ddp = torch.distributed.is_initialized()

    model = model_factory(local_rank=local_rank).to(device)
    model_copy = copy.deepcopy(model)

    sub_model = RNNTSubModels.from_RNNT(model_copy, ddp=ddp, loss_fn=loss_fn)
    if ddp:
        model = DDP(model, device_ids=[torch.cuda.current_device()])
        args.num_gpus = torch.distributed.get_world_size()

    args.no_amp = no_amp
    scaler1 = torch.amp.GradScaler("cuda") if not no_amp else None
    # when running distributed the data on each rank should be different
    data1 = input_data_factory(local_rank=local_rank)
    scaler2, data2 = (copy.deepcopy(scaler1), copy.deepcopy(data1))

    return (sub_model, scaler2, data2), (model, scaler1, data1)


@pytest.fixture()
@torch.no_grad()
def ragged_input_data_factory(input_dim_fixture, n_classes_fixture, device):
    """Per-utterance lengths that differ, so the lattice splits come out uneven."""

    def _gen_data(seed=0, local_rank=0):
        set_seed(seed=seed, local_rank=local_rank)

        feat_lens = [24, 8, 16, 20, 12, 4, 28, 10]
        txt_lens = [12, 3, 7, 9, 5, 1, 14, 4]
        batch_size = len(feat_lens)

        feats = torch.randn(
            max(feat_lens), batch_size, input_dim_fixture, device=device
        )
        txt = torch.randint(
            0, n_classes_fixture - 1, (batch_size, max(txt_lens)), device=device
        )

        return (
            TimeFirst(feats, torch.tensor(feat_lens, device=device, dtype=torch.int)),
            BatchFirst(txt, torch.tensor(txt_lens, device=device, dtype=torch.int)).T,
        )

    return _gen_data


@pytest.mark.parametrize(
    "factory_name,joint_lattice_cap,no_amp",
    [
        ("input_data_factory", CELLS_PER_UTT, True),
        pytest.param(
            "input_data_factory",
            CELLS_PER_UTT,
            False,
            marks=pytest.mark.xfail(reason="amp introduces numerical differences"),
        ),
        ("ragged_input_data_factory", 1, True),
        ("ragged_input_data_factory", 100, True),
        ("ragged_input_data_factory", NO_SPLIT, True),
    ],
)
def test_batch_split_train_step_equiv(
    request,
    model_factory,
    loss_fn,
    args,
    device,
    factory_name,
    joint_lattice_cap,
    no_amp,
):
    """train_step must match reference_train_step for any joint_lattice_cap."""
    args.joint_lattice_cap = joint_lattice_cap
    (model1, scaler1, data1), (model2, scaler2, data2) = build_objects(
        model_factory,
        request.getfixturevalue(factory_name),
        args,
        device,
        no_amp,
        loss_fn,
    )

    results1 = train_step(
        model1,
        args,
        *data1,
        prefix=None,
        scaler=scaler1,
        rnnt_state=None,
        loss_mods=IDENTITY_LOSS_MODIFIERS,
        is_last_microbatch=True,
    )

    results2 = reference_train_step(
        model2,
        loss_fn,
        args,
        *data2,
        prefix=None,
        scaler=scaler2,
        rnnt_state=None,
        loss_mods=IDENTITY_LOSS_MODIFIERS,
        is_last_microbatch=True,
    )

    compare_step_results(results1, results2, model1, model2, scaler1, scaler2)


@pytest.mark.parametrize("conditional", [False, True])
def test_batch_split_train_step_conditional(
    model_factory,
    loss_fn,
    input_data_factory,
    args,
    device,
    conditional,
    n_classes_fixture,
):
    """Test forward passes for train_step with conditional training."""
    args.conditional = conditional

    (model1, scaler1, data1), (model2, scaler2, data2) = build_objects(
        model_factory, input_data_factory, args, device, True, loss_fn
    )
    prefix = None
    if conditional:
        _, tokens = data1
        prefix = TimeFirst(
            data=torch.randint(
                0,
                n_classes_fixture - 1,
                (2, tokens.data.shape[1]),
                device=device,
            ),
            lens=torch.full((tokens.data.shape[1],), 2, device=device, dtype=torch.int),
        )

    results = train_step(
        model1,
        args,
        *data1,
        scaler=scaler1,
        rnnt_state=None,
        loss_mods=IDENTITY_LOSS_MODIFIERS,
        prefix=prefix,
        is_last_microbatch=True,
    )

    assert results[0] >= 0  # loss is non-negative

    results2 = reference_train_step(
        model2,
        loss_fn,
        args,
        *data2,
        scaler=scaler2,
        rnnt_state=None,
        loss_mods=IDENTITY_LOSS_MODIFIERS,
        prefix=prefix,
        is_last_microbatch=True,
    )

    compare_step_results(results, results2, model1, model2, scaler1, scaler2)


def test_batch_split_train_step_conditional_encoder(
    conditional_encoder_model_factory,
    loss_fn,
    input_data_factory,
    args,
    device,
    n_classes_fixture,
):
    """
    Test conditional training when prefixes condition both encoder and predictor.
    """
    args.conditional = True

    (model1, scaler1, data1), (model2, scaler2, data2) = build_objects(
        conditional_encoder_model_factory,
        input_data_factory,
        args,
        device,
        True,
        loss_fn,
    )
    _, tokens = data1
    prefix = TimeFirst(
        data=torch.randint(
            0,
            n_classes_fixture - 1,
            (2, tokens.data.shape[1]),
            device=device,
        ),
        lens=torch.full((tokens.data.shape[1],), 2, device=device, dtype=torch.int),
    )

    results1 = train_step(
        model1,
        args,
        *data1,
        scaler=scaler1,
        rnnt_state=None,
        loss_mods=IDENTITY_LOSS_MODIFIERS,
        prefix=prefix,
        is_last_microbatch=True,
    )

    results2 = reference_train_step(
        model2,
        loss_fn,
        args,
        *data2,
        scaler=scaler2,
        rnnt_state=None,
        loss_mods=IDENTITY_LOSS_MODIFIERS,
        prefix=prefix,
        is_last_microbatch=True,
    )

    compare_step_results(results1, results2, model1, model2, scaler1, scaler2)


@pytest.mark.parametrize("poison", [float("nan"), float("inf")])
def test_batch_split_flags_non_finite_loss(
    model_factory, loss_fn, input_data_factory, args, device, poison
):
    """A non-finite (NaN or inf) loss in ANY split must be reported to the train
    loop, which then drops the whole accumulation window."""
    (sub_model, scaler, data), _ = build_objects(
        model_factory, input_data_factory, args, device, no_amp=True, loss_fn=loss_fn
    )

    calls = {"n": 0}
    orig_forward = loss_fn.forward

    def poisoned_forward(*fwd_args, **fwd_kwargs):
        loss = orig_forward(*fwd_args, **fwd_kwargs)
        calls["n"] += 1
        if calls["n"] == 1:
            return loss * 0 + poison
        return loss

    loss_fn.forward = poisoned_forward

    _, loss_invalid, _ = train_step(
        sub_model,
        args,
        *data,
        prefix=None,
        scaler=scaler,
        rnnt_state=None,
        loss_mods=IDENTITY_LOSS_MODIFIERS,
        is_last_microbatch=True,
    )

    assert calls["n"] == BATCH_SIZE, "every split must still run"
    assert loss_invalid is True


@pytest.mark.parametrize(
    "factory_name", ["model_factory", "conditional_encoder_model_factory"]
)
def test_batch_split_bridge_delivers_encoder_prediction_grads(
    request, factory_name, loss_fn, input_data_factory, args, device
):
    """The ``no_grad`` detach bridge must deliver gradient to the encoder & prediction.

    ``detach_to_bf`` copies the encoder/prediction outputs ``f``/``g`` into fresh fp32
    leaves (``f_2``/``g_2``) inside a ``torch.no_grad()`` block. That deliberately severs
    ``f_2``/``g_2`` from the encoder/prediction autograd graph, so the ONLY route back to
    the encoder & prediction *parameters* is the manual ``f.data.backward(f_2.grad)`` /
    ``g.data.backward(g_2.grad)`` that runs AFTER the split loop. (The joint parameters,
    by contrast, get their gradient directly inside the loop.)

    Hence a broken bridge has a precise, silent symptom: the joint keeps getting
    gradients while every encoder/prediction parameter's ``.grad`` is ``None`` or all
    zero. This asserts the opposite -- each encoder & prediction parameter receives a
    finite, non-zero gradient -- catching that regression directly, with no reference
    model. Runs on a single GPU in the normal suite, so it guards CI even without
    ``torchrun`` (``test_batch_split_bridge_grads_all_reduced_correctly`` extends the
    same check to the multi-GPU all-reduce).
    """
    model_factory = request.getfixturevalue(factory_name)
    args.conditional = factory_name == "conditional_encoder_model_factory"

    (model, _scaler, data), _ref = build_objects(
        model_factory, input_data_factory, args, device, no_amp=True, loss_fn=loss_fn
    )

    prefix = None
    if args.conditional:
        _, tokens = data
        prefix = TimeFirst(
            data=torch.randint(0, 99, (2, tokens.data.shape[1]), device=device),
            lens=torch.full((tokens.data.shape[1],), 2, device=device, dtype=torch.int),
        )

    train_step(
        model,
        args,
        *data,
        scaler=None,
        rnnt_state=None,
        loss_mods=IDENTITY_LOSS_MODIFIERS,
        prefix=prefix,
        is_last_microbatch=True,
    )

    enc_ids = _view_param_ids(model.encoder)
    pred_ids = _view_param_ids(model.prediction)
    checked = {"encoder": 0, "prediction": 0}
    for name, p in model.named_parameters():
        if id(p) in enc_ids:
            group = "encoder"
        elif id(p) in pred_ids:
            group = "prediction"
        else:
            continue  # joint params get grad directly in the loop, not via the bridge
        assert p.grad is not None, (
            f"{group} param {name!r} received NO gradient -- the no_grad detach in "
            f"detach_to_bf was never reconnected by f.data.backward / g.data.backward"
        )
        assert torch.isfinite(
            p.grad
        ).all(), f"{group} param {name!r} has a non-finite gradient"
        assert p.grad.abs().sum().item() > 0.0, (
            f"{group} param {name!r} has an all-zero gradient -- the no_grad bridge did "
            f"not carry the joint-loss gradient back to it"
        )
        checked[group] += 1

    assert (
        checked["encoder"] > 0 and checked["prediction"] > 0
    ), f"expected encoder & prediction params to exist; checked={checked}"


@pytest.mark.parametrize(
    "factory_name", ["model_factory", "conditional_encoder_model_factory"]
)
def test_submodels_partition_all_parameters(request, factory_name, loss_fn):
    """The Encoder/Prediction/JointLoss DDP views must own an exact PARTITION of the
    RNNT's parameters -- every parameter in exactly one view.

    Each view registers only its own sub-modules (sharing the RNNT's parameter tensors),
    so each DDP reduces a disjoint, fully-used set. A parameter in NO view would never be
    all-reduced (silent multi-GPU gradient desync); a parameter in two views would be
    all-reduced twice. This guards the partition against drift -- e.g. a parameter added
    to RNNT but not mapped into a view.
    """
    model = request.getfixturevalue(factory_name)(local_rank=0)
    subs = RNNTSubModels.from_RNNT(model, ddp=False, loss_fn=loss_fn)

    rnnt_ids = {id(p) for p in model.parameters()}
    views = {
        "encoder": {id(p) for p in subs.encoder.parameters()},
        "prediction": {id(p) for p in subs.prediction.parameters()},
        "joint": {id(p) for p in subs.joint_loss.parameters()},
    }

    union = set().union(*views.values())
    assert union == rnnt_ids, (
        f"views do not partition RNNT params: {len(rnnt_ids - union)} unowned "
        f"(never all-reduced -> silent desync), {len(union - rnnt_ids)} foreign"
    )
    assert views["encoder"].isdisjoint(views["prediction"]), "enc/pred share params"
    assert views["encoder"].isdisjoint(views["joint"]), "enc/joint share params"
    assert views["prediction"].isdisjoint(views["joint"]), "pred/joint share params"


@pytest.mark.parametrize(
    "make_output",
    [
        lambda t: TimeFirst(data=t, lens=torch.tensor([t.shape[0]] * t.shape[1])),
        lambda t: Hidden(hidd=t, cell=t),
        lambda t: StateEncoder(pre_stack=Hidden(t, t), aft_stack=Hidden(t, t)),
        lambda t: StatePred(state=Hidden(t, t), embed=t),
        # The full encode() / predict() return tuples that DDP actually traverses:
        lambda t: (
            TimeFirst(data=t, lens=torch.tensor([t.shape[0]] * t.shape[1])),
            StateEncoder(pre_stack=Hidden(t, t), aft_stack=Hidden(t, t)),
        ),
        lambda t: (
            TimeFirst(data=t, lens=torch.tensor([t.shape[0]] * t.shape[1])),
            StatePred(state=Hidden(t, t), embed=t),
        ),
    ],
    ids=[
        "TimeFirst",
        "Hidden",
        "StateEncoder",
        "StatePred",
        "encode_output",
        "predict_output",
    ],
)
def test_ddp_can_locate_grad_tensors_in_forward_output(make_output):
    """
    CPU regression guard for a silent multi-GPU gradient-desync bug.

    DDP arms gradient all-reduce by flattening the wrapped module's forward output
    with ``torch.utils._pytree.tree_flatten`` (see ``DDP._post_forward`` /
    ``_tree_flatten_with_rref``). ``tree_flatten`` only decomposes *registered*
    container types; an unregistered custom ``@dataclass`` is treated as a single
    opaque leaf, so a grad-requiring tensor nested inside it is invisible to DDP. The
    encoder/prediction sub-models return their features and state wrapped in
    ``TimeFirst`` / ``StateEncoder`` / ``StatePred`` / ``Hidden`` dataclasses; if those
    are not registered as pytree nodes, DDP cannot find the feature tensors and
    SILENTLY SKIPS their all-reduce -> encoder/prediction params desync across GPUs ->
    training diverges with no error (single-GPU is unaffected because DDP is not
    involved). Refs: pytorch/pytorch#106690, huggingface/transformers#25357.

    Unlike ``test_dist_equiv_optim_steps`` (which only runs under ``torchrun``), this
    test is CPU-only and runs in the normal suite, so the regression is caught in CI.
    """
    from torch.utils._pytree import tree_flatten

    t = torch.randn(3, 2, 4, requires_grad=True)
    leaves, _ = tree_flatten(make_output(t))
    assert any(leaf is t for leaf in leaves), (
        "DDP-wrapped forward output is not pytree-traversable: a grad-requiring "
        "tensor is hidden inside an unregistered dataclass, so DDP cannot locate it "
        "and will silently skip gradient all-reduce on multi-GPU. Register the batch/"
        "state dataclasses as pytree nodes (torch.utils._pytree.register_pytree_node)."
    )


ALLOW_FAIL_200 = pytest.param(
    200,
    marks=pytest.mark.xfail(
        # Numerical-stability stress case: the tiny random model can NaN past ~150
        # steps. Whether it NaNs is config/seed dependent (e.g. the conditional-encoder
        # variant stays stable), so this is non-strict -- xfail or xpass are both fine.
        reason="NaNs in the loss/weights can occur when n_steps > 150",
        strict=False,
    ),
)


@pytest.mark.parametrize("conditional_encoder", [False, True])
@pytest.mark.parametrize("train_step_fn", [reference_train_step, train_step])
@pytest.mark.parametrize(
    "n_steps",
    [1, 10, ALLOW_FAIL_200],
)
def test_dist_equiv_optim_steps(
    request,
    optimizer_factory,
    loss_fn,
    input_data_factory,
    args,
    device,
    n_classes_fixture,
    train_step_fn,
    n_steps,
    conditional_encoder,
):
    if not torch.distributed.is_initialized():
        pytest.skip("Skipping distributed test")
    local_rank = torch.distributed.get_rank()
    args.no_amp = True
    # conditional_encoder exercises the multilingual prod path: a per-utterance prefix
    # conditions both the encoder and the prediction network.
    args.conditional = conditional_encoder
    model_factory = request.getfixturevalue(
        "conditional_encoder_model_factory" if conditional_encoder else "model_factory"
    )

    model = model_factory(local_rank=local_rank).to(device)
    optimizer = optimizer_factory(model)
    if train_step_fn == reference_train_step:
        model = DDP(model, device_ids=[torch.cuda.current_device()])
    else:
        model = RNNTSubModels.from_RNNT(model, ddp=True, loss_fn=loss_fn)

    args.num_gpus = torch.distributed.get_world_size()

    # after using DDP the model weights should be the same across ranks
    assert_model_equal_on_all_ranks(model, local_rank, check_grads=False)

    for i in range(n_steps):
        # The data (and prefix) differ per rank; DDP must still all-reduce the grads so
        # that they are identical across ranks afterwards.
        input_data = input_data_factory(seed=i * 10, local_rank=local_rank)
        prefix = None
        if conditional_encoder:
            _, tokens = input_data
            batch_size = tokens.data.shape[1]
            prefix = TimeFirst(
                data=torch.randint(
                    0, n_classes_fixture - 1, (2, batch_size), device=device
                ),
                lens=torch.full((batch_size,), 2, device=device, dtype=torch.int),
            )
        optimizer.zero_grad()
        _, loss_nan, _ = _run_step(
            train_step_fn,
            model,
            loss_fn,
            args,
            *input_data,
            scaler=None,
            rnnt_state=None,
            loss_mods=IDENTITY_LOSS_MODIFIERS,
            prefix=prefix,
            is_last_microbatch=True,
        )
        if loss_nan:
            continue
        assert_model_equal_on_all_ranks(model, local_rank, check_grads=True)
        optimizer.step()

    assert_model_equal_on_all_ranks(model, local_rank, check_grads=False)


@pytest.mark.parametrize("train_step_fn", [reference_train_step, train_step])
def test_dist_no_sync_accumulation(
    model_factory,
    optimizer_factory,
    loss_fn,
    input_data_factory,
    args,
    device,
    train_step_fn,
):
    """Multi-micro-batch gradient accumulation with no_sync.

    Runs one micro-batch with ``is_last_microbatch=False`` (DDP ``no_sync`` -- gradients
    accumulate locally, no all-reduce) followed by one with ``is_last_microbatch=True``
    (all-reduce). Exercises the is_last_microbatch=False path end-to-end (it must not
    error, e.g. "marked ready twice") and checks the accumulated gradients are identical
    across ranks after the final synchronising micro-batch.
    """
    if not torch.distributed.is_initialized():
        pytest.skip("Skipping distributed test")
    local_rank = torch.distributed.get_rank()
    args.no_amp = True
    model = model_factory(local_rank=local_rank).to(device)
    # Build the optimizer on the bare RNNT (it owns the params) before wrapping.
    optimizer = optimizer_factory(model)
    if train_step_fn == reference_train_step:
        model = DDP(model, device_ids=[torch.cuda.current_device()])
    else:
        model = RNNTSubModels.from_RNNT(model, ddp=True, loss_fn=loss_fn)
    args.num_gpus = torch.distributed.get_world_size()

    optimizer.zero_grad()

    n_microbatches = 2
    for mb in range(n_microbatches):
        data = input_data_factory(seed=mb * 10, local_rank=local_rank)
        _run_step(
            train_step_fn,
            model,
            loss_fn,
            args,
            *data,
            scaler=None,
            rnnt_state=None,
            loss_mods=IDENTITY_LOSS_MODIFIERS,
            prefix=None,
            is_last_microbatch=(mb + 1 == n_microbatches),
        )

    # The final (sync) micro-batch all-reduces the accumulated gradients, so every rank
    # must now hold identical gradients.
    assert_model_equal_on_all_ranks(model, local_rank, check_grads=True)


@pytest.mark.parametrize("conditional_encoder", [False, True])
def test_batch_split_syncs_once_per_optimizer_step(
    request,
    optimizer_factory,
    loss_fn,
    input_data_factory,
    args,
    device,
    n_classes_fixture,
    conditional_encoder,
):
    """Each component DDP must all-reduce exactly once per optimizer step.

    A comm hook on the encoder/prediction/joint DDPs counts bucket all-reduces. Over
    ``n`` accumulation micro-batches (``is_last_microbatch=False`` for the first n-1,
    ``True`` on the last), every component must reduce ZERO times on the no_sync
    micro-batches and >=1 time on the final one -- including the encoder/prediction
    gradients produced by the delayed ``f.data.backward`` / ``g.data.backward`` *after*
    the split loop, outside any sync_ctx. This is the direct guard that wrapping only
    the forward in ``no_sync`` (not the trailing backward) still suppresses all-reduce:
    DDP binds the sync decision to forward time, so the later backward inherits it.

    The hook still performs the real all-reduce, so on >=2 ranks the post-step cross-rank
    grad check confirms the synced gradient is correct. The count assertions are
    world_size-independent (the comm hook fires whenever the reducer is armed), so this
    is a live guard even on a single GPU.
    """
    if not torch.distributed.is_initialized():
        pytest.skip("Skipping distributed test")
    from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook

    local_rank = torch.distributed.get_rank()
    args.no_amp = True
    args.conditional = conditional_encoder
    args.num_gpus = torch.distributed.get_world_size()
    model_factory = request.getfixturevalue(
        "conditional_encoder_model_factory" if conditional_encoder else "model_factory"
    )
    model = model_factory(local_rank=local_rank).to(device)
    optimizer = optimizer_factory(model)
    model = RNNTSubModels.from_RNNT(model, ddp=True, loss_fn=loss_fn)

    # Count bucket all-reduces per component. The hook still performs the real
    # all-reduce, so gradients stay correct (and the cross-rank check below is valid).
    pg = torch.distributed.group.WORLD
    counts = {"encoder": 0, "prediction": 0, "joint": 0}

    def make_hook(name):
        def hook(state, bucket):
            counts[name] += 1
            return allreduce_hook(state, bucket)

        return hook

    model.encoder.register_comm_hook(pg, make_hook("encoder"))
    model.prediction.register_comm_hook(pg, make_hook("prediction"))
    model.joint_loss.register_comm_hook(pg, make_hook("joint"))

    optimizer.zero_grad()
    n_microbatches = 3
    for mb in range(n_microbatches):
        is_last = mb + 1 == n_microbatches
        prefix = None
        data = input_data_factory(seed=mb * 10, local_rank=local_rank)
        if conditional_encoder:
            _, tokens = data
            batch_size = tokens.data.shape[1]
            prefix = TimeFirst(
                data=torch.randint(
                    0, n_classes_fixture - 1, (2, batch_size), device=device
                ),
                lens=torch.full((batch_size,), 2, device=device, dtype=torch.int),
            )

        before = dict(counts)
        _, loss_nan, _ = train_step(
            model,
            args,
            *data,
            scaler=None,
            rnnt_state=None,
            loss_mods=IDENTITY_LOSS_MODIFIERS,
            prefix=prefix,
            is_last_microbatch=is_last,
        )
        assert not loss_nan, f"unexpected NaN on micro-batch {mb}"
        delta = {k: counts[k] - before[k] for k in counts}

        if is_last:
            assert all(
                v >= 1 for v in delta.values()
            ), f"the final micro-batch must all-reduce every component, got {delta}"
        else:
            assert all(v == 0 for v in delta.values()), (
                f"no_sync micro-batch {mb} all-reduced {delta}; the encoder/prediction "
                "backward after the split loop must inherit no_sync from its forward"
            )

    # On >=2 ranks the per-rank data differs, so identical grads here prove the final
    # all-reduce both ran and produced the correct synchronised gradient.
    assert_model_equal_on_all_ranks(model, local_rank, check_grads=True)


@pytest.mark.parametrize("conditional_encoder", [False, True])
def test_batch_split_bridge_grads_all_reduced_correctly(
    request,
    loss_fn,
    input_data_factory,
    args,
    device,
    n_classes_fixture,
    conditional_encoder,
):
    """The encoder/prediction (``no_grad``-bridge) gradients must be all-reduced to the
    exact cross-rank MEAN -- not dropped, not left at this rank's local value.

    This is the direct multi-GPU test of the worry that the ``torch.no_grad()`` detach in
    ``detach_to_bf`` -- together with the trailing ``f.data.backward`` / ``g.data.backward``
    that runs OUTSIDE any ``sync_ctx`` -- might exclude the encoder/prediction gradients
    from DDP's all-reduce. ``test_batch_split_syncs_once_per_optimizer_step`` proves those
    all-reduces *fire*; this proves they reduce the *correct value*.

    Method (one DDP-wrapped RNNTSubModels, per-rank-distinct data):
      1. LOCAL pass with ``is_last_microbatch=False`` -> every component runs under
         ``no_sync`` so NOTHING is all-reduced. Snapshot each encoder/prediction
         parameter's purely-local gradient.
      2. Manually all-reduce (sum / world_size) copies of those snapshots: the mean a
         correct DDP all-reduce must produce.
      3. Re-seed identical data, zero grads, then a SYNC pass with
         ``is_last_microbatch=True`` -> DDP all-reduces. Dropout is 0.0 and the LSTM is
         deterministic, so the local contribution is identical to step 1.
      4. Post-sync grad must equal the manual mean for every bridge parameter. Were the
         bridge grads skipped by all-reduce they would instead keep this rank's local
         value (which differs across ranks), failing the check. On >=2 ranks we also
         assert the synced grad MOVED off the local value, proving the all-reduce really
         touched these params (and that the per-rank data genuinely differed).
    """
    if not torch.distributed.is_initialized():
        pytest.skip("Skipping distributed test")
    world_size = torch.distributed.get_world_size()
    local_rank = torch.distributed.get_rank()
    args.no_amp = True
    args.num_gpus = world_size
    args.conditional = conditional_encoder
    model_factory = request.getfixturevalue(
        "conditional_encoder_model_factory" if conditional_encoder else "model_factory"
    )
    model = model_factory(local_rank=local_rank).to(device)
    model = RNNTSubModels.from_RNNT(model, ddp=True, loss_fn=loss_fn)

    bridge_ids = _view_param_ids(model.encoder) | _view_param_ids(model.prediction)
    bridge = [(n, p) for n, p in model.named_parameters() if id(p) in bridge_ids]
    assert bridge, "no encoder/prediction params found"

    def run(is_last):
        # Re-seed identical per-rank data before each pass so the two passes are bitwise
        # deterministic (dropout is 0.0), then run from a clean grad slate.
        data = input_data_factory(seed=0, local_rank=local_rank)
        prefix = None
        if conditional_encoder:
            _, tokens = data
            batch_size = tokens.data.shape[1]
            prefix = TimeFirst(
                data=torch.randint(
                    0, n_classes_fixture - 1, (2, batch_size), device=device
                ),
                lens=torch.full((batch_size,), 2, device=device, dtype=torch.int),
            )
        for _n, p in model.named_parameters():
            p.grad = None
        train_step(
            model,
            args,
            *data,
            scaler=None,
            rnnt_state=None,
            loss_mods=IDENTITY_LOSS_MODIFIERS,
            prefix=prefix,
            is_last_microbatch=is_last,
        )

    # 1. Purely-local gradients: is_last_microbatch=False -> no component all-reduces.
    run(is_last=False)
    local = {n: p.grad.detach().clone() for n, p in bridge}

    # 2. The cross-rank mean a correct all-reduce must produce. sorted() gives an
    #    identical iteration order on every rank, so these collectives line up.
    expected = {}
    for n in sorted(local):
        g = local[n].clone()
        torch.distributed.all_reduce(g, op=torch.distributed.ReduceOp.SUM)
        g /= world_size
        expected[n] = g

    # 3. SYNC pass on the same data -> DDP all-reduces the bridge grads (the backward for
    #    these params runs after the split loop, outside any sync_ctx).
    run(is_last=True)
    synced = {n: p.grad.detach().clone() for n, p in bridge}

    # 4. Every bridge grad must equal the manual mean.
    bad = [
        n
        for n in local
        if not torch.allclose(synced[n], expected[n], rtol=1e-4, atol=1e-5)
    ]
    assert not bad, (
        f"{len(bad)}/{len(local)} encoder/prediction params were NOT all-reduced to the "
        f"cross-rank mean -- the no_grad bridge / post-loop backward dropped them from "
        f"DDP's all-reduce (a silent multi-GPU gradient desync). First few: {bad[:5]}"
    )

    if world_size >= 2:
        moved = [
            n
            for n in local
            if not torch.allclose(synced[n], local[n], rtol=1e-4, atol=1e-5)
        ]
        assert moved, (
            "synced grads equal this rank's LOCAL grads on every bridge param; with "
            "per-rank-distinct data a real all-reduce must change them. The bridge "
            "grads' all-reduce silently no-opped (or the per-rank data did not differ)."
        )


# (train_step_fn, loss_kind, n_classes, joint_n_hid) for
# test_ddp_matches_single_process_full_batch. The fused kernels need dims the conftest
# defaults do not divide, hence 128/64 there.
DDP_FULL_BATCH_CASES = [
    pytest.param(reference_train_step, "unfused", 100, 13, id="unfused-whole-batch"),
    pytest.param(train_step, "unfused", 100, 13, id="unfused-batch-split"),
    pytest.param(train_step, "fused", 128, 64, id="fused-batch-split"),
]


@pytest.mark.parametrize("n_microbatches", [1, 2])
@pytest.mark.parametrize(
    "train_step_fn,loss_kind,n_classes_fixture,joint_n_hid_fixture",
    DDP_FULL_BATCH_CASES,
    indirect=["n_classes_fixture", "joint_n_hid_fixture"],
)
def test_ddp_matches_single_process_full_batch(
    model_factory,
    input_data_factory,
    args,
    device,
    train_step_fn,
    loss_kind,
    n_classes_fixture,
    joint_n_hid_fixture,
    n_microbatches,
):
    """N-GPU DDP == ONE process over the concatenated full batch -- across accumulation.

    This is the property that *justifies* data-parallel training, and the one the other
    distributed tests here do NOT check. They assert the ranks agree with EACH OTHER after
    the all-reduce (``test_dist_equiv_optim_steps``) and that the all-reduce produces the
    mean of the per-rank grads (``test_batch_split_bridge_grads_all_reduced_correctly``) --
    but a bug that perturbs every rank IDENTICALLY (a wrong loss normalisation, a missing
    ``/world_size``, a factor-of-N) passes all of those while still disagreeing with the
    true single-GPU gradient. This pins the DDP result to that single-process ground truth.

    With ``n_microbatches == 2`` it also covers gradient accumulation with ``no_sync``: the
    first micro-batch runs with ``is_last_microbatch=False`` so every component is under
    ``no_sync`` -- grads accumulate locally with NO all-reduce -- and only the second
    (``is_last_microbatch=True``) all-reduces. The reduction therefore acts on the SUM of
    both micro-batches, so the test confirms the *accumulated* gradient (not just one step)
    all-reduces to exactly the single-process accumulation over the same two full batches.
    Grads are never zeroed between micro-batches.

    The transducer loss is mean-reduced over the batch (``loss.py``) and DDP averages grads
    across ranks, so N ranks of B samples is EXACTLY one process over the N*B batch: the
    per-rank ``1/B`` and the cross-rank ``1/N`` compose into ``1/(N*B)``, with no leftover
    factor-of-N. Verified directly:
      * DDP path: wrap the model, accumulate ``train_step_fn`` over the micro-batches (only
        the last syncs) -> all-reduce of the accumulated grad -> ``g_ddp`` (identical on
        every rank).
      * Reference: a bare copy of the same initial weights, accumulated over the full N*B
        batch per micro-batch, rebuilt from every rank's (deterministic) data, with
        ``num_gpus=1`` so it performs no collectives.
    Then ``g_ddp == g_full`` for every parameter.

    Parametrised over the whole-batch, batch-split and fused-batch-split DDP paths, so it
    also certifies batch-split multi-GPU == single-GPU. The fused joint+loss has no
    whole-batch reference implementation, so its reference is the same fused strategy at
    ``batch_split_factor=1``: that additionally pins split-and-accumulate == one
    whole-batch call (``loss_divisor = GAB * BSF`` must compose back to ``GAB`` at BSF=1).
    A BSF that divides the batch of 8 evenly is required for that identity.

    Non-conditional only: concatenating per-utterance prefixes is out of scope here; the
    conditional path's cross-rank consistency is covered by ``test_dist_equiv_optim_steps``.
    """
    if not torch.distributed.is_initialized():
        pytest.skip("Skipping distributed test")
    world_size = torch.distributed.get_world_size()
    local_rank = torch.distributed.get_rank()
    args.no_amp = True
    args.conditional = False

    fused = loss_kind == "fused"
    if fused:
        loss_fn = build_fused_loss(n_classes_fixture, joint_n_hid_fixture)
        args.batch_split_factor = 4
        # The fused kernels accumulate in a different order, so looser tolerances.
        rtol, atol = 2e-2, 2e-3
    else:
        loss_fn = build_unfused_loss(n_classes_fixture)
        rtol, atol = 1e-3, 1e-4

    # Identical initial weights on every rank (seed 0, rank-independent); snapshot a bare
    # reference copy BEFORE the DDP wrap so it shares those exact starting weights.
    base = model_factory(local_rank=0).to(device)
    ref = copy.deepcopy(base)
    if train_step_fn == reference_train_step:
        ddp_model = DDP(base, device_ids=[torch.cuda.current_device()])
    else:
        ddp_model = RNNTSubModels.from_RNNT(base, ddp=True, loss_fn=loss_fn)

    # One distinct micro-batch of data per accumulation step.
    seeds = list(range(n_microbatches))

    # DDP path: accumulate over micro-batches WITHOUT zeroing grads; only the final
    # micro-batch syncs, so any earlier one runs under no_sync and the all-reduce acts on
    # the accumulated sum -> g_ddp (identical on every rank).
    args.num_gpus = world_size
    for m, seed in enumerate(seeds):
        rank_feats, rank_tokens = input_data_factory(seed=seed, local_rank=local_rank)
        _run_step(
            train_step_fn,
            ddp_model,
            loss_fn,
            args,
            rank_feats,
            rank_tokens,
            scaler=None,
            rnnt_state=None,
            loss_mods=IDENTITY_LOSS_MODIFIERS,
            prefix=None,
            is_last_microbatch=(m == n_microbatches - 1),
        )

    # Reference: the SAME accumulation over the full N*B batch per micro-batch, rebuilt
    # locally from every rank's deterministic data (input_data_factory is a pure function
    # of (seed, local_rank)), run single-process. is_last_microbatch is a no-op for the
    # bare (non-DDP) model; the accumulation comes from not zeroing between micro-batches.
    ref_args = copy.deepcopy(args)
    ref_args.num_gpus = 1  # bare single process: no collectives in is_loss_invalid
    if fused:
        ref_args.batch_split_factor = 1
        ref_model = RNNTSubModels.from_RNNT(ref, ddp=False, loss_fn=loss_fn)
        ref_step_fn = train_step
    else:
        ref_model = ref
        ref_step_fn = reference_train_step

    for m, seed in enumerate(seeds):
        feats_parts, tokens_parts = [], []
        for r in range(world_size):
            f, t = input_data_factory(seed=seed, local_rank=r)
            feats_parts.append(f)
            tokens_parts.append(t)
        _run_step(
            ref_step_fn,
            ref_model,
            loss_fn,
            ref_args,
            _cat_time_first(feats_parts),
            _cat_time_first(tokens_parts),
            scaler=None,
            rnnt_state=None,
            loss_mods=IDENTITY_LOSS_MODIFIERS,
            prefix=None,
            is_last_microbatch=(m == n_microbatches - 1),
        )

    # g_ddp (on the shared RNNT params) must equal the full-batch single-process grad.
    src = unwrap_ddp(ddp_model) if isinstance(ddp_model, DDP) else ddp_model
    ddp_params = dict(src.named_parameters())
    bad = []
    for name, p_ref in ref_model.named_parameters():
        p_ddp = ddp_params[name]
        assert p_ddp.grad is not None, f"{name}: DDP grad is None"
        assert p_ref.grad is not None, f"{name}: single-process grad is None"
        if not torch.allclose(p_ddp.grad, p_ref.grad, rtol=rtol, atol=atol):
            bad.append(name)
    assert not bad, (
        f"{len(bad)}/{len(ddp_params)} params: {world_size}-GPU {loss_kind} DDP gradient "
        f"!= single-process full-batch gradient over {n_microbatches} accumulation "
        f"step(s) -- data-parallel training is NOT equivalent to single-GPU (e.g. wrong "
        f"loss normalisation / missing cross-rank average). First few: {bad[:5]}"
    )


def assert_model_equal_on_all_ranks(
    model: Union[DDP, RNNTSubModels], local_rank, check_grads
):
    """
    Assert the model weights or grads are the same across all ranks.

    All ranks >0 send their model tensors to rank 0, which compares them. The pass/fail
    verdict is then reduced across ranks and asserted on every rank, so a real mismatch
    fails cleanly everywhere instead of deadlocking -- otherwise rank 0 would raise while
    the other ranks block forever on the following collective.
    """
    params_wrong = []
    if local_rank == 0:
        for rank in range(1, torch.distributed.get_world_size()):
            params_wrong.extend(receive_and_compare_model(model, rank, check_grads))
    else:
        send_model(model, local_rank, check_grads)

    # all_reduce is a collective that every rank reaches, so no rank can raise (and
    # abandon later collectives) while another waits. It both synchronises and carries
    # the verdict to every rank.
    num_wrong = torch.tensor(
        [len(params_wrong)], dtype=torch.long, device=torch.cuda.current_device()
    )
    torch.distributed.all_reduce(num_wrong)
    assert num_wrong.item() == 0, (
        f"{check_grads=}: {num_wrong.item()} param(s) differ across ranks "
        f"(rank 0 names: {params_wrong}). DDP did not synchronise these parameters -- "
        f"e.g. forward outputs not pytree-traversable so all-reduce was skipped."
    )


def send_model(model, local_rank, check_grads):
    for _name, param in model.named_parameters():
        send = param if not check_grads else param.grad
        torch.distributed.send(send, dst=0, tag=local_rank)


def receive_and_compare_model(model, rank, check_grads):
    params_wrong = []
    for name, param in model.named_parameters():
        local = param if not check_grads else param.grad
        received_param = torch.empty_like(local)
        torch.distributed.recv(received_param, src=rank, tag=rank)
        if not torch.allclose(local, received_param, atol=1e-4, rtol=1e-3):
            print(f"Rank {rank} param mismatch ({check_grads=}): {name}")
            params_wrong.append(name)
    return params_wrong


def compare_step_results(
    results1,
    results2,
    model1: RNNTSubModels,
    model2: Union[RNNT, DDP],
    scaler1,
    scaler2,
):
    """
    Assert that results1 and results2 are approximately equal.
    """
    loss_item1, loss_nan1, rnnt_state1 = results1
    loss_item2, loss_nan2, rnnt_state2 = results2

    assert pytest.approx(loss_item1) == loss_item2
    assert loss_nan1 == loss_nan2

    model2 = unwrap_ddp(model2)
    params2 = {k: v for k, v in model2.named_parameters()}
    grad_failed = []
    for k, v in model1.named_parameters():
        assert torch.allclose(v, params2[k])
        if scaler1 is not None:
            assert scaler1.get_scale() == scaler2.get_scale()
        try:
            # fill inf grads with 0
            mask_inf = v.grad.isinf() | params2[k].grad.isinf()
            v.grad.masked_fill_(mask_inf, 0)
            params2[k].grad.masked_fill_(mask_inf, 0)

            assert torch.allclose(
                v.grad, params2[k].grad, rtol=1e-4, atol=1e-4
            ), f"{k} grad mismatch"
        except AssertionError:
            grad_failed.append(k)
    assert (
        not grad_failed
    ), f"Grad check failed for {len(grad_failed)} params: {grad_failed}"


@pytest.mark.parametrize("joint_lattice_cap", [1, 100])
def test_val_loss_lattice_split_equiv(
    model_factory,
    loss_fn,
    ragged_input_data_factory,
    args,
    device,
    joint_lattice_cap,
):
    """``--joint_lattice_cap`` must not change the validation loss."""
    model = model_factory().to(device)
    sub = RNNTSubModels.from_RNNT(model, ddp=False, loss_fn=loss_fn)
    feats, tokens = ragged_input_data_factory()

    split = val_loss(
        model,
        sub.joint_loss,
        feats=feats,
        tokens=tokens,
        loss_mods=IDENTITY_LOSS_MODIFIERS,
        prefix=None,
        joint_lattice_cap=joint_lattice_cap,
    )
    whole = val_loss(
        model,
        sub.joint_loss,
        feats=feats,
        tokens=tokens,
        loss_mods=IDENTITY_LOSS_MODIFIERS,
        prefix=None,
        joint_lattice_cap=None,
    )

    torch.testing.assert_close(split, whole)


@pytest.mark.parametrize(
    "cells,cap,expected",
    [
        ([1, 1, 1], 10, [(0, 3)]),
        ([4, 4, 4], 8, [(0, 2), (2, 3)]),
        ([4, 4], 8, [(0, 2)]),
        ([99], 8, [(0, 1)]),
        ([99, 1], 8, [(0, 1), (1, 2)]),
    ],
)
def test_get_bsplit(cells, cap, expected):
    """A single utterance over the cap gets its own split, and exceeds it."""
    assert get_bsplit(cells, cap) == expected


if __name__ == "__main__":
    import os
    from argparse import ArgumentParser

    import torch.distributed.run as distrib_run

    parser = ArgumentParser()
    parser.add_argument("--called_by_torchrun", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument(
        "--local_rank",
        default=os.getenv("LOCAL_RANK", 0),
        type=int,
        help="GPU id used for distributed training",
    )

    parsed_args = parser.parse_args()

    if not parsed_args.called_by_torchrun:
        torchrun_parser = distrib_run.get_args_parser()
        torchrun_args = torchrun_parser.parse_args(
            [
                "--standalone",
                "--nnodes",
                "1",
                "--nproc_per_node",
                str(parsed_args.num_gpus),
                "tests/rnnt/test_batch_split.py",
                "--called_by_torchrun",
            ]
        )
        distrib_run.run(torchrun_args)
        exit()

    # We are running with torchrun, so we should run the tests
    # and exit
    local_rank = os.getenv("LOCAL_RANK", 0)
    torch.cuda.set_device(parsed_args.local_rank)

    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
    )
    pytest.main([__file__])
    exit()
