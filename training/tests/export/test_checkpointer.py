import collections
import os

import pytest
import torch


def compare_models(model1, model2, equal=True, epsilon=1e-6):
    """
    Compare parameters of two models.

    :param model1: The first model
    :param model2: The second model
    :param equal: If True assert models equal, else assert models unequal.
    :param epsilon: Tolerance for comparing weights
    :raises AssertionError: if models have different parameters
    """
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if equal:
            assert torch.allclose(p1, p2, atol=epsilon)
        else:
            assert not torch.allclose(p1, p2, atol=epsilon)


@pytest.mark.parametrize("epoch", [1, 10])
@pytest.mark.parametrize("seed", [1, 2])
def test_save_load_model(
    checkpointer_tmp_dir,
    mini_model_factory,
    optimizer_factory,
    tokenizer_kw,
    tmp_path,
    epoch,
    seed,
    logmel_norm_weight=1.0,
):
    step = 100
    best_wer = 3.05
    meta = {"best_wer": best_wer, "start_epoch": epoch, "step": step}
    meta2 = {"best_wer": 10000, "start_epoch": 10000, "step": 10000}

    # Generate two models with different seeds
    model, _ = mini_model_factory(seed=seed)
    model_2, _ = mini_model_factory(seed=3)
    optimizer = optimizer_factory(model)

    # Assert model weights are different before loading weights
    compare_models(model, model_2, equal=False)
    assert meta != meta2

    # Save model weights (temporarily)
    checkpointer_tmp_dir.save(
        model,
        None,
        optimizer,
        epoch,
        step,
        best_wer,
        tokenizer_kw,
        logmel_norm_weight=logmel_norm_weight,
    )

    # Load weights
    fpath = os.path.join(tmp_path, f"RNN-T_step{step}_checkpoint.pt")
    checkpointer_tmp_dir.load(fpath, model_2, None, meta=meta2)

    # Assert model weights are equal after loading weights
    compare_models(model, model_2)
    assert meta == meta2


def test_checkpointer_structure(
    checkpointer_tmp_dir,
    mini_model_factory,
    optimizer_factory,
    tokenizer_kw,
    logmel_norm_weight=1.0,
):
    epoch = 5
    step = 100
    best_wer = 3.05

    # Generate two models
    model, _ = mini_model_factory()
    ema_model, _ = mini_model_factory()
    optimizer = optimizer_factory(model)

    # Save model state (temporarily)
    checkpointer_tmp_dir.save(
        model,
        ema_model,
        optimizer,
        epoch,
        step,
        best_wer,
        tokenizer_kw,
        logmel_norm_weight=logmel_norm_weight,
    )

    # Load checkpoint
    last_ckpt = checkpointer_tmp_dir.last_checkpoint()
    state = torch.load(last_ckpt)

    # Assert all keys are present and no extras
    expected_keys = [
        "epoch",
        "step",
        "best_wer",
        "state_dict",
        "ema_state_dict",
        "optimizer",
        "tokenizer_kw",
        "logmel_norm_weight",
    ]
    assert set(state.keys()) == set(
        expected_keys
    ), "Checkpoint keys do not match expected keys."

    # Assert type of each value in loaded state
    assert isinstance(state["epoch"], int)
    assert isinstance(state["step"], int)
    assert isinstance(state["best_wer"], float)
    assert isinstance(state["state_dict"], collections.OrderedDict)
    assert isinstance(state["ema_state_dict"], collections.OrderedDict)
    assert isinstance(state["optimizer"], dict)
    assert isinstance(state["tokenizer_kw"], dict)
    assert isinstance(state["logmel_norm_weight"], float)
