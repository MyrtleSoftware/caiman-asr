# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# val.py is mostly just train.py with most of the training code removed.
# rob@myrtle, May 2022

import os
from argparse import ArgumentParser, Namespace

import torch
from beartype.typing import Any, Dict

from rnnt_train.common.evaluate import evaluate
from rnnt_train.common.helpers import print_once
from rnnt_train.common.shared_args import add_shared_args, check_shared_args
from rnnt_train.common.tb_dllogger import flush_log
from rnnt_train.setup.base import VAL, BuiltObjects
from rnnt_train.setup.val import ValSetup


def val_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="RNN-T Training Reference")

    training = parser.add_argument_group("training/validation setup")
    training.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        default=True,
        help="Enable cudnn benchmark",
    )
    training.add_argument(
        "--amp",
        "--fp16",
        action="store_true",
        default=False,
        help="Use mixed precision",
    )
    training.add_argument("--seed", default=None, type=int, help="Random seed")
    training.add_argument(
        "--local_rank",
        default=os.getenv("LOCAL_RANK", 0),
        type=int,
        help="GPU id used for distributed processing",
    )

    optim = parser.add_argument_group("optimization setup")
    optim.add_argument(
        "--val_batch_size", default=2, type=int, help="Evaluation time batch size"
    )

    io = parser.add_argument_group("feature and checkpointing setup")
    io.add_argument(
        "--dali_device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Use DALI pipeline for fast data processing",
    )
    io.add_argument(
        "--ckpt",
        default=None,
        type=str,
        required=True,
        help="Path to the checkpoint to use",
    )
    io.add_argument(
        "--model_config",
        default="configs/testing-1023sp_run.yaml",
        type=str,
        required=True,
        help="Path of the model configuration file",
    )
    io.add_argument(
        "--val_manifests",
        type=str,
        required=False,
        default=None,
        nargs="+",
        help="Paths of the evaluation datasets manifest files. "
        "Ignored if --read_from_tar=True",
    )
    io.add_argument(
        "--read_from_tar",
        action="store_true",
        default=False,
        help="Read data from tar files instead of json manifest files",
    )
    io.add_argument(
        "--val_tar_files",
        type=str,
        required=False,
        default=None,
        nargs="+",
        help="Paths of the evaluation dataset tar files. "
        "Ignored if --read_from_tar=False.",
    )
    io.add_argument(
        "--dataset_dir", required=True, type=str, help="Root dir of dataset"
    )
    io.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for logs and checkpoints",
    )
    io.add_argument(
        "--log_file", type=str, default=None, help="Path to save the logfile."
    )
    io.add_argument(
        "--skip_init",
        action="store_true",
        default=False,
        help="If true do not re-initialise things that should only be intialised once",
    )
    io.add_argument(
        "--max_symbol_per_sample",
        type=int,
        default=None,
        help="maximum number of symbols per sample can have during eval",
    )
    io.add_argument(
        "--no_loss",
        action="store_true",
        help="To use less memory, don't calculate transducer loss",
    )
    add_shared_args(parser)
    return parser


def validate(
    args: Namespace, val_objects: BuiltObjects, return_dataloader: bool = False
) -> Dict[str, Any]:
    """
    Validates model on dataset in args.val_manifests or args.val_tar_files.

    If return_dataloader=True, return dataloader in results dict.
    """
    if val_objects.multi_gpu:
        torch.distributed.barrier()

    # A checkpoint should always be specified
    assert args.ckpt is not None

    epoch = 1
    step = None  # Switches off logging of val data results to TensorBoard

    val_loader = val_objects.data_objects[VAL].loader
    results = evaluate(
        epoch,
        step,
        val_loader,
        val_objects.feat_procs[VAL],
        val_objects.tokenizers[VAL].detokenize,
        val_objects.ema_model,
        val_objects.loss_fn,
        val_objects.decoder,
        args,
        calculate_loss=not args.no_loss,
    )
    wer = results["wer"]
    flush_log()
    val_files = args.val_manifests if not args.read_from_tar else args.val_tar_files
    val_files_str = " ".join(val_files)
    if len(val_files_str) > 100:
        val_files_str = val_files_str[:100] + "..."
    print_once(f'\nWord Error Rate: {wer*100.0:5.3f}% on "{val_files_str}"\n')

    if return_dataloader:
        results["dataloader"] = val_loader

    return results


def main(args, val_objects):
    check_shared_args(args)
    # check data path args
    if not args.read_from_tar:
        assert (
            args.val_manifests is not None
        ), "Must provide val_manifests if not reading from tar"

    validate(args, val_objects=val_objects)


if __name__ == "__main__":
    parser = val_arg_parser()
    args = parser.parse_args()
    val_objects = ValSetup().run(args)
    main(args, val_objects)
