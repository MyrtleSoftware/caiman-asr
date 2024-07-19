from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from tests.export.test_checkpointer import compare_models

from caiman_asr_train.export.hardware_ckpt import create_hardware_ckpt


@pytest.fixture()
def mock_args(test_data_dir, tmp_path):
    step = 100

    args = {
        "ckpt": str(tmp_path / f"RNN-T_step{step}_checkpoint.pt"),
        "melmeans": str(test_data_dir / "melmeans.pt"),
        "melvars": str(test_data_dir / "melvars.pt"),
        "melalpha": 0.0,
        "output_ckpt": str(test_data_dir / "hardware_ckpt.pt"),
        "skip_ngram": False,
        "override_ngram_path": None,
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
        model,
        model,
        optimizer,
        epoch,
        step,
        best_wer,
        tokenizer_kw,
        logmel_norm_weight=1.0,
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
        "ngram",
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
    assert set(hardcp.keys()) == set(
        loaded_hardcp.keys()
    ), "Hardware checkpoint keys do not match keys from loaded hardware checkpoint."
    assert hardcp["epoch"] == loaded_hardcp["epoch"]
    assert hardcp["step"] == loaded_hardcp["step"]
    assert hardcp["best_wer"] == loaded_hardcp["best_wer"]
    assert torch.allclose(hardcp["melmeans"], loaded_hardcp["melmeans"])
    assert torch.allclose(hardcp["melvars"], loaded_hardcp["melvars"])
    assert hardcp["sentpiece_model"] == loaded_hardcp["sentpiece_model"]
    assert hardcp["ngram"]["binary"] == loaded_hardcp["ngram"]["binary"]
    assert hardcp["ngram"]["scale_factor"] == loaded_hardcp["ngram"]["scale_factor"]
    assert isinstance(hardcp["ngram"]["binary"], bytes)

    # the window size changes depending on the model size so exclude it from comparison
    hardcp["rnnt_config"]["input_val"]["filterbank_features"].pop("window_size")
    loaded_hardcp["rnnt_config"]["input_val"]["filterbank_features"].pop("window_size")
    try:
        assert hardcp["rnnt_config"] == loaded_hardcp["rnnt_config"]
    except AssertionError as err:
        msg = str(err)
        msg += f"""\nIf this check is failing, the yaml config fields don't match the
        ones in the hardware checkpoint in {mock_args.output_ckpt}.
        If you have added or removed a field, please note that this is liable to break
        the downstream inference server. Many additions to the yaml configs aren't needed
        at inference time as they are training-specific. If this applies to your case
        you can add the field to the _allow_ignore set in the relevant Schema class and
        this test will pass unchanged.
        If your change doesn't fall under the above exception (and it was intentional),
        please edit the test_data hardware checkpoint in order to make this test pass
        and then ensure that the Myrtle inference server is updated to expect your
        change.
        """
        raise AssertionError(msg) from err


def test_export_raises_ramp(
    mini_model_factory,
    optimizer_factory,
    config_fp,
    checkpointer_tmp_dir,
    tokenizer_kw,
    mock_args,
):
    """
    check that hardware export raises an error if the ramp period is not complete.
    """
    epoch = 10
    step = 100

    config_fp = str(Path(__file__).parent.parent.parent / config_fp)
    # Generate model
    model, model_cfg_fp = mini_model_factory(config_fp, seed=1)
    optimizer = optimizer_factory(model)

    # Save model checkpoint (temporarily)
    checkpointer_tmp_dir.save(
        model,
        model,
        optimizer,
        epoch,
        step,
        0.05,
        tokenizer_kw,
        logmel_norm_weight=0.5,
    )

    mock_args.config = model_cfg_fp

    with pytest.raises(AssertionError):
        _ = create_hardware_ckpt(mock_args)
