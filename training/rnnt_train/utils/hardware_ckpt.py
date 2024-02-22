#!/usr/bin/env python3

import argparse
from argparse import Namespace

import torch

from rnnt_train.rnnt import config
from rnnt_train.rnnt.config_schema import RNNTInferenceConfigSchema
from rnnt_train.utils.model_schema import check_model_schema, return_schemas


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
        "--melmeans", type=str, default="/results/melmeans.pt", help="mel means file"
    )
    parser.add_argument(
        "--melvars", type=str, default="/results/melvars.pt", help="mel vars file"
    )
    parser.add_argument(
        "--melalpha",
        type=float,
        default=0.001,
        help="streaming normalization time constant",
    )
    parser.add_argument(
        "--output_ckpt",
        type=str,
        default="/results/hardware_ckpt.pt",
        help="name of the output hardware checkpoint file",
    )
    args = parser.parse_args()
    return args


def load_checkpoint(fpath):
    """Load training checkpoint."""
    return torch.load(fpath, map_location="cpu")


def load_mel_stats(fpath_means, fpath_vars):
    """Load the mel stats."""
    return torch.load(fpath_means, map_location="cpu"), torch.load(
        fpath_vars, map_location="cpu"
    )


def extract_sentencepiece_fname(config):
    """Extract the sentencepiece model filename from the config file."""
    with open(config, "r") as f:
        for line in f:
            ll = line.split()
            if len(ll) > 0 and ll[0] == "sentpiece_model:":
                return ll[1]
    return False


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

    melmeans, melvars = load_mel_stats(args.melmeans, args.melvars)
    melalpha = args.melalpha

    spmfn = extract_sentencepiece_fname(args.config)
    assert spmfn

    spmb = read_sentencepiece_model(spmfn)

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
        "version": "1.8.0",  # add the semantic version number of the hardware checkpoint
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
