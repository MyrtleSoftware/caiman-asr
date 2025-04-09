import pytest
import torch
import torch.nn as nn
from tests.export.test_checkpointer import compare_models

from caiman_asr_train.export.checkpoint_averaging import average_checkpoints


@pytest.fixture()
def simple_linear_model():
    def create_model(weights):
        model = nn.Linear(2, 2, bias=False)
        model.weight = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        return model

    return create_model


def save_model_ckpt(model, path):
    torch.save({"state_dict": model.state_dict()}, path)


def test_average_single_checkpoint(mini_model_factory, tmp_path):
    """Test that averaging single checkpoint returns same weights."""
    checkpoint_path = tmp_path / "model.pt"

    model, _ = mini_model_factory()
    save_model_ckpt(model, checkpoint_path)

    avg_state_dict, _ = average_checkpoints([checkpoint_path])
    model2, _ = mini_model_factory()
    model2.load_state_dict(avg_state_dict)

    compare_models(model, model2)


@pytest.mark.parametrize(
    "weights, exp_weights",
    [
        ([[[1, 1], [1, 1]], [[2, 2], [2, 2]]], [[1.5, 1.5], [1.5, 1.5]]),
        ([[[1, 0], [10, 2]], [[1, 1], [2, 4]]], [[1, 0.5], [6, 3]]),
        ([[[1, 1], [1, 1]], [[2, 2], [2, 2]]], [[1.5, 1.5], [1.5, 1.5]]),
        ([[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]], [[2, 2], [2, 2]]),
        ([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], [[1, 1], [1, 1]]),
        ([[[1, 1], [1, 1]], [[1.0, 1.0], [1.0, 1.0]]], [[1, 1], [1, 1]]),
    ],
)
def test_average_checkpoints(simple_linear_model, tmp_path, weights, exp_weights):
    """Test averaging two or more simple model checkpoints."""
    checkpoint_paths = []
    # Load weights into nn.Linear and save to temp file
    for i, weight in enumerate(weights):
        model = simple_linear_model(weight)
        checkpoint_path = tmp_path / f"model{i}.pt"
        save_model_ckpt(model, checkpoint_path)
        checkpoint_paths.append(checkpoint_path)

    # Pass list of checkpoint (temp) paths.
    # average_checkpoints() loads all checkpoints in list, averages them,
    # and returns state dict with averaged weights
    avg_weights, _ = average_checkpoints(checkpoint_paths)
    expected_weights = torch.tensor(exp_weights, dtype=torch.float32)
    assert torch.allclose(avg_weights["weight"].to(torch.float32), expected_weights)
