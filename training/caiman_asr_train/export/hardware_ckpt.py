#!/usr/bin/env python3

import argparse
import math
from argparse import Namespace
from pathlib import Path

import torch
from beartype.typing import Tuple

from caiman_asr_train.export.config_schema import RNNTInferenceConfigSchema
from caiman_asr_train.export.model_schema import check_model_schema, return_schemas
from caiman_asr_train.rnnt import config


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser(
        description="Gather training results into a hardware checkpoint"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/results/RNN-T_best_checkpoint.pt",
        help="checkpoint file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/testing-1023sp_run.yaml",
        help="config file",
    )
    parser.add_argument(
        "--melalpha",
        type=float,
        default=0.0,
        help="DEPRECATED: streaming normalization time constant. Streaming normalization "
        "is no longer used in the asr-server so this value should always be 0.0 (meaning "
        "the mel stats are always used as-is for normalization without a time-decay).",
    )
    parser.add_argument(
        "--output_ckpt",
        type=str,
        default="/results/hardware_ckpt.pt",
        help="name of the output hardware checkpoint file",
    )
    args = parser.parse_args()

    assert args.melalpha == 0.0, "melalpha is deprecated and should be 0.0."
    return args


def load_checkpoint(fpath):
    """Load training checkpoint."""
    return torch.load(fpath, map_location="cpu")


def load_stats_from_disk(fpath_means, fpath_vars):
    return torch.load(fpath_means, map_location="cpu"), torch.load(
        fpath_vars, map_location="cpu"
    )


def load_mel_stats(
    args: Namespace, train_cfg: dict, ckpt: dict
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Load mel stats.
    """
    logmel_norm_weight = ckpt["logmel_norm_weight"]
    assert math.isclose(logmel_norm_weight, 1.0), (
        f"logmel_norm_weight should be 1.0 but it is {logmel_norm_weight}. When this "
        "value is less than 1.0, the ramp period did not complete during training.\nThis "
        f"means WER could be improved as the trained model at '{args.ckpt}' does not "
        "expect NormType.DATASET_STATS at inference time.\n\n"
        "Please --resume training from this checkpoint to some --training_steps greater "
        "than --norm_ramp_end_step and then run this script again with the newly trained "
        "checkpoint."
    )
    (_, features_kw, _, _) = config.input(train_cfg, "val")
    stats_dir = Path(features_kw["stats_path"])
    assert stats_dir.exists(), f"Stats directory {stats_dir} does not exist."
    melmeans, melvars = load_stats_from_disk(
        stats_dir / "melmeans.pt", stats_dir / "melvars.pt"
    )
    return melmeans, melvars, args.melalpha


def read_sentencepiece_model(fname):
    """Read the sentencepiece model file into a bytes object."""
    with open(fname, "rb") as f:
        return f.read()


def inference_only_config(config_fp: str) -> dict:
    """Return the minimal config required to run inference.

    Training-specific parameters are removed from the config.
    """
    config_dict = config.load(config_fp)
    inference_only_cfg = RNNTInferenceConfigSchema(**config_dict)

    return inference_only_cfg.dict()


def create_hardware_ckpt(args) -> dict:
    traincp = load_checkpoint(args.ckpt)
    train_cfg = config.load(args.config)
    melmeans, melvars, melalpha = load_mel_stats(args, train_cfg, traincp)

    spm_fn = train_cfg["tokenizer"]["sentpiece_model"]
    assert spm_fn, "Sentencepiece model file not found in config."
    spmb = read_sentencepiece_model(spm_fn)

    inference_config = inference_only_config(args.config)
    # It is useful to be able to run these hardware checkpoints using val.py in Python.
    # So ema_state_dict is renamed to state_dict in the hardware checkpoint which
    # val.py will load with warnings.
    hardcp = {
        "state_dict": traincp["ema_state_dict"],
        "epoch": traincp["epoch"],
        "step": traincp["step"],
        "best_wer": traincp["best_wer"],
        "melmeans": melmeans,
        "melvars": melvars,
        "melalpha": melalpha,
        "sentpiece_model": spmb,  # store the bytes object in the hardware checkpoint
        "version": "1.9.0",  # add the semantic version number of the hardware checkpoint
        "rnnt_config": inference_config,  # copy in inference config
    }

    return hardcp


def save_hardware_ckpt(hardcp, output_ckpt):
    torch.save(hardcp, output_ckpt)


def main():
    args = parse_arguments()
    hardcp = create_hardware_ckpt(args)
    schemas = return_schemas()
    check_model_schema(hardcp["state_dict"], schemas)
    save_hardware_ckpt(hardcp, args.output_ckpt)


if __name__ == "__main__":
    main()
