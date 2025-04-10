import collections
import os
import pathlib

import pytest
import torch

from caiman_asr_train.export.checkpointer import Checkpointer


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
        config_path="won't_be_used.txt",
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
        config_path="won't_be_used.txt",
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


def test_tracked(
    mini_model_factory,
    optimizer_factory,
):
    tmp_3000 = "/tmp/rnnt_step3000_checkpoint.pt"
    tmp_4000 = "/tmp/rnnt_step4000_checkpoint.pt"
    pathlib.Path(tmp_3000).touch()
    pathlib.Path(tmp_4000).touch()
    cp = Checkpointer(save_dir="/tmp", model_name="rnnt")
    assert len(cp.tracked) == 2
    assert cp.tracked == {
        3000: "/tmp/rnnt_step3000_checkpoint.pt",
        4000: "/tmp/rnnt_step4000_checkpoint.pt",
    }
    #
    model, _ = mini_model_factory(seed=1)
    optimizer = optimizer_factory(model)
    cp.save(model, None, optimizer, 5, 5000, 10, {}, 1.0, "won't_be_used.txt")
    assert len(cp.tracked) == 3
    os.remove(tmp_3000)
    os.remove(tmp_4000)
    os.remove("/tmp/rnnt_step5000_checkpoint.pt")
