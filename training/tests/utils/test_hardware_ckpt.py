from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from tests.common.test_helpers import compare_models

from rnnt_train.utils.hardware_ckpt import create_hardware_ckpt


@pytest.fixture()
def mock_args(test_data_dir, tmp_path):
    epoch = 10

    args = {
        "ckpt": str(tmp_path / f"RNN-T_epoch{epoch}_checkpoint.pt"),
        "melmeans": str(test_data_dir / "melmeans.pt"),
        "melvars": str(test_data_dir / "melvars.pt"),
        "melalpha": 0.001,
        "output_ckpt": str(test_data_dir / "hardware_ckpt.pt"),
    }

    return MagicMock(**args)


@pytest.mark.parametrize(
    "config_fp",
    [
        "configs/testing-1023sp_run.yaml",
        "configs/base-8703sp.yaml",
        "configs/large-17407sp.yaml",
    ],
)
def test_create_hardware_ckpt(
    mini_model_factory,
    optimizer_factory,
    config_fp,
    checkpointer_tmp_dir,
    tokenizer_kw,
    mock_args,
):
    epoch = 10
    step = 100
    best_wer = 3.05

    config_fp = str(Path(__file__).parent.parent.parent / config_fp)
    # Generate model
    model, model_cfg_fp = mini_model_factory(config_fp, seed=1)
    optimizer = optimizer_factory(model)

    # Save model checkpoint (temporarily)
    checkpointer_tmp_dir.save(
        model, model, optimizer, epoch, step, best_wer, tokenizer_kw
    )

    # Generate hardware checkpoint from newly saved model checkpoint
    mock_args.config = model_cfg_fp
    hardcp = create_hardware_ckpt(mock_args)

    # Validate structure of newly generated hardware checkpoint
    expected_keys = [
        "state_dict",
        "epoch",
        "step",
        "best_wer",
        "melmeans",
        "melvars",
        "melalpha",
        "sentpiece_model",
        "version",
        "rnnt_config",
    ]
    assert set(hardcp.keys()) == set(
        expected_keys
    ), "Hardware checkpoint keys do not match expected keys."

    # Load previously generated hardware checkpoint
    loaded_hardcp = torch.load(mock_args.output_ckpt)

    # Assert same model state dict in the two hardware checkpoints
    model_2, _ = mini_model_factory(config_fp, seed=2)
    model_2.load_state_dict(loaded_hardcp["state_dict"])
    compare_models(model, model_2)

    # Compare contents of loaded hardware checkpoint to newly generated hardware checkpoint
    assert hardcp["epoch"] == loaded_hardcp["epoch"]
    assert hardcp["step"] == loaded_hardcp["step"]
    assert hardcp["best_wer"] == loaded_hardcp["best_wer"]
    assert torch.allclose(hardcp["melmeans"], loaded_hardcp["melmeans"])
    assert torch.allclose(hardcp["melvars"], loaded_hardcp["melvars"])
    assert hardcp["sentpiece_model"] == loaded_hardcp["sentpiece_model"]

    # the window size changes depending on the model size so exclude it from comparison
    hardcp["rnnt_config"]["input_val"]["filterbank_features"].pop("window_size")
    loaded_hardcp["rnnt_config"]["input_val"]["filterbank_features"].pop("window_size")
    assert hardcp["rnnt_config"] == loaded_hardcp["rnnt_config"]
