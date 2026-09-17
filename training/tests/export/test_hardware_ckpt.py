from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from beartype.roar import BeartypeCallHintViolation

from caiman_asr_train.core.load_hw import _load_hw_dict, load_hw_checkpoint
from caiman_asr_train.export.checkpointer import Checkpointer
from caiman_asr_train.export.hardware_ckpt import (
    create_hardware_ckpt,
    save_hardware_ckpt,
)


def _ngram_binary_as_bytes(binary):
    """Normalise either ngram.binary representation (tensor or bytes) to bytes."""
    return binary.numpy().tobytes() if torch.is_tensor(binary) else binary


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
        "configs/testing-1023sp.yaml",
        "configs/base-8703sp.yaml",
        "configs/large-17407sp.yaml",
    ],
)
def test_create_hardware_ckpt(
    mini_model_factory,
    optimizer_factory,
    config_fp,
    checkpointer_tmp_dir: Checkpointer,
    tokenizer_kw,
    mock_args,
):
    epoch = 10
    step = 100
    best_wer = 3.05
    duration = 123.45

    config_fp = str(Path(__file__).parent.parent.parent / config_fp)
    # Generate model
    model, model_cfg_fp = mini_model_factory(config_fp, seed=1)
    optimizer = optimizer_factory(model)

    # Save model checkpoint (temporarily)
    checkpointer_tmp_dir.save(
        model=model,
        ema_model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        duration=duration,
        best_wer=best_wer,
        tokenizer_kw=tokenizer_kw,
        logmel_norm_weight=1.0,
        config_path="won't_be_used.txt",
    )

    # Generate hardware checkpoint from newly saved model checkpoint
    mock_args.config = model_cfg_fp
    hardcp = create_hardware_ckpt(mock_args)

    # Check that current hard ckpt loads
    _load_hw_dict(hardcp)

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
    assert set(hardcp.keys()) - {"raw_prefix_token_map"} == set(
        expected_keys
    ), "Hardware checkpoint keys do not match expected keys."

    # Load previously generated hardware checkpoint, this is a historical
    # "golden" checkpoint
    loaded_hardcp = torch.load(mock_args.output_ckpt, weights_only=False)

    # The old "golden" checkpoint had a bugged configuration, `in_feats` was
    # `5` in the mini config but the mel `n_filt`/`frame_stacking` didn't
    # match. The new mini config patches that issue by using `in_feats = 240`.
    # We patch the "golden" config/state dict:
    loaded_hardcp["rnnt_config"]["rnnt"]["in_feats"] = 240

    patch_state_dict_keys = [
        # Currently only need to patch input->hidden
        "encoder.pre_rnn.lstm.weight_ih_l0",
    ]

    for k in patch_state_dict_keys:
        loaded_hardcp["state_dict"][k] = hardcp["state_dict"][k].detach().clone()

    # Check that the "golden" hardware checkpoint loads after patching
    _load_hw_dict(loaded_hardcp)

    # Assert correct behaviour loading older hardware ckpt
    model_2, _ = mini_model_factory(config_fp, seed=2)
    loaded_state_dict_copy = deepcopy(loaded_hardcp["state_dict"])
    missing, unexpected = model_2.load_state_dict(loaded_state_dict_copy, strict=False)
    assert missing == ["prefix_token_map", "encoder.prefix_embd.weight"]
    assert unexpected == []

    # Assert correct behaviour loading older hardware ckpt after patching
    # missing keys
    model_3, _ = mini_model_factory(config_fp, seed=3)
    loaded_state_dict_copy = deepcopy(loaded_hardcp["state_dict"])
    # Write new entries to older hardware ckpt
    for k in ["prefix_token_map", "encoder.prefix_embd.weight"]:
        loaded_state_dict_copy[k] = hardcp["state_dict"][k]
    model_3.load_state_dict(loaded_state_dict_copy)
    # Can't test for equality with old state dict because random matrix
    # initialisation has changed with shape of `in_feats`

    # Compare contents of loaded hardware checkpoint to newly generated hardware checkpoint
    assert set(hardcp.keys()) - {"raw_prefix_token_map"} == set(
        loaded_hardcp.keys()
    ), "Hardware checkpoint keys do not match keys from loaded hardware checkpoint."
    assert hardcp["epoch"] == loaded_hardcp["epoch"]
    assert hardcp["step"] == loaded_hardcp["step"]
    assert hardcp["best_wer"] == loaded_hardcp["best_wer"]
    assert torch.allclose(hardcp["melmeans"], loaded_hardcp["melmeans"])
    assert torch.allclose(hardcp["melvars"], loaded_hardcp["melvars"])
    assert hardcp["sentpiece_model"] == loaded_hardcp["sentpiece_model"]
    # The "golden" checkpoint predates the tensor format, so compare content
    # regardless of representation.
    assert _ngram_binary_as_bytes(hardcp["ngram"]["binary"]) == _ngram_binary_as_bytes(
        loaded_hardcp["ngram"]["binary"]
    )
    assert hardcp["ngram"]["scale_factor"] == loaded_hardcp["ngram"]["scale_factor"]
    assert isinstance(hardcp["ngram"]["binary"], torch.Tensor)
    assert isinstance(loaded_hardcp["ngram"]["binary"], bytes)

    # the window size changes depending on the model size so exclude it from comparison
    hardcp["rnnt_config"]["input_val"]["filterbank_features"].pop("window_size")
    loaded_hardcp["rnnt_config"]["input_val"]["filterbank_features"].pop("window_size")

    # The tokenisers character set can evolve overt time so we don't check it
    hardcp["rnnt_config"]["tokenizer"]["labels"] = loaded_hardcp["rnnt_config"][
        "tokenizer"
    ]["labels"]

    # append_EOI is an optional key, if missing it is assumed False
    if "append_EOI" in hardcp["rnnt_config"]["input_val"]:
        hardcp["rnnt_config"]["input_val"].pop("append_EOI")

    # Older hardware checkpoints does not support conditional_encoder. Allow the
    # legacy fixture to omit it, but still compare it when both sides include it.
    if "conditional_encoder" not in loaded_hardcp["rnnt_config"]["rnnt"]:
        hardcp["rnnt_config"]["rnnt"].pop("conditional_encoder", None)

    try:
        assert hardcp["rnnt_config"] == loaded_hardcp["rnnt_config"]
    except AssertionError as err:
        msg = f"""\n
        If this check is failing, the yaml config fields don't match the ones
        in the hardware checkpoint in {mock_args.output_ckpt}. If you have
        added or removed a field, please note that this is liable to break the
        downstream inference server. Many additions to the yaml configs aren't
        needed at inference time as they are training-specific. If this applies
        to your case you can add the field to the allow_ignore set in the
        relevant Schema class and this test will pass unchanged. If your change
        doesn't fall under the above exception (and it was intentional), please
        edit the test_data hardware checkpoint in order to make this test pass
        and then ensure that the Myrtle inference server is updated to expect
        your change.
        """

        # Pretty print the differences
        in_hardcp = set(hardcp["rnnt_config"].keys())
        in_loaded = set(loaded_hardcp["rnnt_config"].keys())

        diff = in_hardcp.symmetric_difference(in_loaded)

        if diff:
            msg += f"\n\nDifferences found in rnnt_config keys: {diff}"
        else:
            for key in hardcp["rnnt_config"]:
                lhs = hardcp["rnnt_config"][key]
                rhs = loaded_hardcp["rnnt_config"][key]
                if lhs != rhs:
                    msg += f"\n\nMismatch in key '{key}': {lhs} != {rhs}"

        raise AssertionError(msg) from err


@pytest.mark.parametrize(
    "config_fp",
    [
        "configs/testing-1023sp.yaml",
        "configs/base-8703sp.yaml",
        "configs/large-17407sp.yaml",
    ],
)
def test_load_hw_checkpoint_round_trips_weights(
    mini_model_factory,
    optimizer_factory,
    config_fp,
    checkpointer_tmp_dir: Checkpointer,
    tokenizer_kw,
    mock_args,
    tmp_path,
):
    """
    End-to-end check that a hardware checkpoint written to disk and read back
    through the public `load_hw_checkpoint` entry point yields a model holding
    the checkpoint's weights.

    This guards against a regression where the loader returns a freshly
    constructed (randomly initialised) model and silently discards the loaded
    state dict.
    """
    epoch = 10
    step = 100
    best_wer = 3.05
    duration = 123.45

    config_fp = str(Path(__file__).parent.parent.parent / config_fp)
    model, model_cfg_fp = mini_model_factory(config_fp, seed=1)
    optimizer = optimizer_factory(model)

    checkpointer_tmp_dir.save(
        model=model,
        ema_model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        duration=duration,
        best_wer=best_wer,
        tokenizer_kw=tokenizer_kw,
        logmel_norm_weight=1.0,
        config_path="won't_be_used.txt",
    )

    mock_args.config = model_cfg_fp
    hardcp = create_hardware_ckpt(mock_args)

    # Write the checkpoint out and read it back via the public API so the whole
    # disk -> load_hw_checkpoint -> reconstructed model path is exercised.
    out_path = tmp_path / "hardware_ckpt.pt"
    save_hardware_ckpt(hardcp, out_path)
    loaded = load_hw_checkpoint(out_path)

    # The reconstructed model must carry the checkpoint's weights. If the loader
    # returned a fresh, random model the keys would still line up but the tensor
    # values would differ, so compare them element-wise.
    loaded_state = loaded.model.state_dict()
    assert loaded_state.keys() == hardcp["state_dict"].keys()
    for key, expected in hardcp["state_dict"].items():
        assert torch.equal(loaded_state[key], expected), (
            f"Weight '{key}' in the loaded model does not match the checkpoint; "
            "the loader is discarding the loaded state dict."
        )

    # The remaining fields should also be threaded through unchanged.
    assert loaded.version == hardcp["version"]
    assert loaded.sentpiece_model_binary == hardcp["sentpiece_model"]
    assert loaded.ngram is not None
    assert isinstance(loaded.ngram.binary, torch.Tensor)
    assert torch.equal(loaded.ngram.binary, hardcp["ngram"]["binary"])
    assert loaded.ngram.scale == hardcp["ngram"]["scale_factor"]
    assert torch.equal(loaded.mel_stats.means, hardcp["melmeans"])
    assert torch.equal(loaded.mel_stats.vars, hardcp["melvars"])
    assert loaded.mel_stats.alpha == hardcp["melalpha"]
    # Only the mau model compiler writes a program, so a checkpoint straight
    # out of the export path has none.
    assert loaded.mau_program is None


@pytest.mark.parametrize("legacy_style", ["null", "absent"])
def test_load_hw_checkpoint_recovers_legacy_prefix_token_map(
    mini_model_factory,
    optimizer_factory,
    config_fp,
    checkpointer_tmp_dir: Checkpointer,
    tokenizer_kw,
    mock_args,
    legacy_style,
):
    """
    Older conditional-encoder checkpoints carry no usable prefix token map at
    the top level: some have `raw_prefix_token_map` set to None, and some omit
    the key. Only the state dict's `prefix_token_map` lookup buffer holds the
    map, so it has to be recovered by inverting that.

    Loading these used to fail outright, because the loader built a
    non-conditional model and then hit unexpected keys in the state dict.
    """
    config_fp = str(Path(__file__).parent.parent.parent / config_fp)
    model, model_cfg_fp = mini_model_factory(config_fp, seed=1)
    optimizer = optimizer_factory(model)

    checkpointer_tmp_dir.save(
        model=model,
        ema_model=model,
        optimizer=optimizer,
        epoch=10,
        step=100,
        duration=123.45,
        best_wer=3.05,
        tokenizer_kw=tokenizer_kw,
        logmel_norm_weight=1.0,
        config_path="won't_be_used.txt",
    )

    mock_args.config = model_cfg_fp
    hardcp = create_hardware_ckpt(mock_args)

    expected = hardcp["raw_prefix_token_map"]
    assert expected, "fixture should be a conditional-encoder model"

    # Degrade the checkpoint to the legacy layout.
    if legacy_style == "null":
        hardcp["raw_prefix_token_map"] = None
    else:
        del hardcp["raw_prefix_token_map"]

    loaded = _load_hw_dict(hardcp)

    assert loaded.model.config.prefix_token_map == expected

    # The conditional modules must exist and hold the checkpoint's values, so
    # the recovered map has to agree with the buffer it was recovered from.
    assert torch.equal(
        loaded.model.prefix_token_map, hardcp["state_dict"]["prefix_token_map"]
    )
    assert torch.equal(
        loaded.model.encoder["prefix_embd"].weight,
        hardcp["state_dict"]["encoder.prefix_embd.weight"],
    )


def test_load_hw_checkpoint_carries_mau_program(
    mini_model_factory,
    optimizer_factory,
    config_fp,
    checkpointer_tmp_dir: Checkpointer,
    tokenizer_kw,
    mock_args,
):
    """
    The mau model compiler injects its compiled program into a hardware
    checkpoint as `mau_program`, turning it into a CAIMAN-ASR checkpoint (see
    `docs/src/inference/compiling_weights.md`).

    Nothing in this repo writes that key, so this is the only guard that the
    loader hands the program back untouched, and that a malformed one is not
    quietly reported as no program at all.
    """
    config_fp = str(Path(__file__).parent.parent.parent / config_fp)
    model, model_cfg_fp = mini_model_factory(config_fp, seed=1)
    optimizer = optimizer_factory(model)

    checkpointer_tmp_dir.save(
        model=model,
        ema_model=model,
        optimizer=optimizer,
        epoch=10,
        step=100,
        duration=123.45,
        best_wer=3.05,
        tokenizer_kw=tokenizer_kw,
        logmel_norm_weight=1.0,
        config_path="won't_be_used.txt",
    )

    mock_args.config = model_cfg_fp
    hardcp = create_hardware_ckpt(mock_args)

    assert "mau_program" not in hardcp, "the export path must not write a program"
    assert _load_hw_dict(hardcp).mau_program is None

    # The program's shape is the compiler's business; it is passed through
    # whatever it holds.
    program = {"instructions": [1, 2, 3], "meta": {"target": "sc250"}}
    hardcp["mau_program"] = program

    assert _load_hw_dict(hardcp).mau_program == program

    hardcp["mau_program"] = "not-a-dict"

    with pytest.raises(BeartypeCallHintViolation):
        _load_hw_dict(hardcp)


def test_export_raises_ramp(
    mini_model_factory,
    optimizer_factory,
    config_fp,
    checkpointer_tmp_dir: Checkpointer,
    tokenizer_kw,
    mock_args,
):
    """
    check that hardware export raises an error if the ramp period is not complete.
    """
    epoch = 10
    step = 100
    duration = 123.45

    config_fp = str(Path(__file__).parent.parent.parent / config_fp)
    # Generate model
    model, model_cfg_fp = mini_model_factory(config_fp, seed=1)
    optimizer = optimizer_factory(model)

    # Save model checkpoint (temporarily)
    checkpointer_tmp_dir.save(
        model=model,
        ema_model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        duration=duration,
        best_wer=0.05,
        tokenizer_kw=tokenizer_kw,
        logmel_norm_weight=0.5,
        config_path="won't_be_used.txt",
    )

    mock_args.config = model_cfg_fp

    with pytest.raises(AssertionError):
        _ = create_hardware_ckpt(mock_args)
