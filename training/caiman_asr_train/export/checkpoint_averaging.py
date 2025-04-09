#!/usr/bin/env python3

import argparse
import copy
import os
from collections import OrderedDict

import torch
from beartype.typing import List, Optional, Tuple

from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.export.checkpointer import Checkpointer
from caiman_asr_train.rnnt import config
from caiman_asr_train.rnnt.model import RNNT


def accumulate_state_dicts(
    state_dict: OrderedDict, sum_state_dict: OrderedDict
) -> None:
    """Accumulates values from one state dictionary into another."""
    for key, value in state_dict.items():
        if key not in sum_state_dict:
            sum_state_dict[key] = value.clone()
        else:
            sum_state_dict[key] += value


def average_checkpoints(
    checkpoint_paths: List[str],
) -> Tuple[OrderedDict, Optional[OrderedDict]]:
    """
    Averages the parameters of the given checkpoints.
    """
    num_ckpts = len(checkpoint_paths)
    sum_model_state_dict = OrderedDict()
    sum_ema_state_dict = OrderedDict()

    ema_present = True

    for ckpt_path in checkpoint_paths:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model_state_dict = checkpoint["state_dict"]
        ema_state_dict = checkpoint.get("ema_state_dict", None)

        accumulate_state_dicts(model_state_dict, sum_model_state_dict)

        if ema_state_dict:
            accumulate_state_dicts(ema_state_dict, sum_ema_state_dict)
        else:
            ema_present = False

    # Divide by the number of checkpoints to get the average
    for key in sum_model_state_dict.keys():
        sum_model_state_dict[key].div_(num_ckpts)

    if ema_present:
        for key in sum_ema_state_dict.keys():
            sum_ema_state_dict[key].div_(num_ckpts)
    else:
        sum_ema_state_dict = None

    return sum_model_state_dict, sum_ema_state_dict


def main(args):
    cfg = config.load(args.model_config)
    tokenizer_kw = config.get_tokenizer_conf(cfg)
    tokenizer = Tokenizer(**tokenizer_kw)
    rnnt_config = config.rnnt(cfg)

    model = RNNT(n_classes=tokenizer.num_labels + 1, **rnnt_config).to("cpu")
    ema_model = copy.deepcopy(model).to("cpu")

    averaged_state_dict, averaged_ema_state_dict = average_checkpoints(args.ckpts)

    model.load_state_dict(averaged_state_dict, strict=True)
    ema_model.load_state_dict(averaged_ema_state_dict, strict=True)

    output_dir, output_name = os.path.split(args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    checkpointer = Checkpointer(output_dir, output_name)
    checkpointer.save(
        model,
        ema_model,
        None,
        0,
        0,
        0,
        tokenizer_kw,
        1.0,
        args.model_config,
        is_best=False,
        filepath=args.output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average Checkpoints")
    parser.add_argument(
        "--ckpts",
        "--checkpoints",
        nargs="+",
        required=True,
        help="List of absolute checkpoint paths to average.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path (.pt) for averaged checkpoint.",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="/workspace/training/configs/testing-1023sp_run.yaml",
        help="Path of the model configuration file.",
    )
    args = parser.parse_args()

    main(args)
