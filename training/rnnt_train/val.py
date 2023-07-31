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

import copy
import os
import random
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from rnnt_train.common.data import features
from rnnt_train.common.data.build_dataloader import build_dali_loader
from rnnt_train.common.data.text import Tokenizer
from rnnt_train.common.evaluate import evaluate
from rnnt_train.common.helpers import Checkpointer, num_weights, print_once
from rnnt_train.common.tb_dllogger import flush_log, init_log, log
from rnnt_train.mlperf import logging
from rnnt_train.rnnt import config
from rnnt_train.rnnt.decoder import RNNTGreedyDecoder
from rnnt_train.rnnt.loss import apexTransducerLoss
from rnnt_train.rnnt.model import RNNT


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
        "--val_batch_size", default=2, type=int, help="Evalution time batch size"
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
        "Ignored if --read_from_tar=False. If not provided,"
        "we use all tar files in the directory specified by --dataset_dir.",
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
        "--max_symbol_per_sample",
        type=int,
        default=None,
        help="maximum number of symbols per sample can have during eval",
    )
    return parser


def validate(args: Namespace):
    """
    Validates model on dataset in args.val_manifests or args.val_tar_files.
    """
    assert torch.cuda.is_available()

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    # set up distributed processing
    multi_gpu = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if multi_gpu:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
        print_once(f"Distributed processing with {world_size} GPUs\n")
    else:
        world_size = 1

    if args.seed is not None:
        logging.log_event(logging.constants.SEED, value=args.seed)
        torch.manual_seed(args.seed + args.local_rank)
        np.random.seed(args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

    init_log(args)

    cfg = config.load(args.model_config)

    logging.log_end(logging.constants.INIT_STOP)
    if multi_gpu:
        torch.distributed.barrier()
    logging.log_start(logging.constants.RUN_START)
    if multi_gpu:
        torch.distributed.barrier()

    print_once("Setting up datasets...")
    (
        val_dataset_kw,
        val_features_kw,
        val_splicing_kw,
        val_specaugm_kw,
    ) = config.input(cfg, "val")

    # A checkpoint should always be specified
    assert args.ckpt is not None

    tokenizer_kw = config.tokenizer(cfg)
    tokenizer = Tokenizer(**tokenizer_kw)
    blank_idx = tokenizer.num_labels

    val_augmentations = torch.nn.Sequential(
        val_specaugm_kw
        and features.SpecAugment(optim_level=args.amp, **val_specaugm_kw)
        or torch.nn.Identity(),
        features.FrameSplicing(optim_level=args.amp, **val_splicing_kw),
        features.PermuteAudio(),
    )

    val_loader = build_dali_loader(
        args,
        "val",
        batch_size=args.val_batch_size,
        dataset_kw=val_dataset_kw,
        features_kw=val_features_kw,
        tokenizer=tokenizer,
    )

    val_feat_proc = val_augmentations

    val_feat_proc.cuda()

    if not args.read_from_tar:
        logging.log_event(logging.constants.EVAL_SAMPLES, value=val_loader.dataset_size)

    # set up the RNNT model
    rnnt_config = config.rnnt(cfg)
    model = RNNT(n_classes=tokenizer.num_labels + 1, **rnnt_config)
    model.cuda()

    print_once(f"Model size: {num_weights(model) / 10**6:.1f}M params\n")

    loss_fn = apexTransducerLoss(blank_idx=blank_idx, packed_input=False)

    logging.log_event(
        logging.constants.EVAL_MAX_PREDICTION_SYMBOLS, value=args.max_symbol_per_sample
    )
    greedy_decoder = RNNTGreedyDecoder(
        blank_idx=blank_idx, max_symbol_per_sample=args.max_symbol_per_sample
    )

    ema_model = copy.deepcopy(model).cuda()

    if multi_gpu:
        model = DistributedDataParallel(model)

    # setup checkpointer
    checkpointer = Checkpointer(args.output_dir, "RNN-T", [])

    # load checkpoint (modified to not need optimizer / meta args)
    checkpointer.load(args.ckpt, model, ema_model)

    epoch = 1
    step = None  # Switches off logging of val data results to TensorBoard

    wer = evaluate(
        epoch,
        step,
        val_loader,
        val_feat_proc,
        tokenizer.detokenize,
        ema_model,
        loss_fn,
        greedy_decoder,
        args,
    )

    flush_log()
    val_files = args.val_manifests if not args.read_from_tar else args.val_tar_files
    val_files_str = " ".join(val_files)
    if len(val_files_str) > 100:
        val_files_str = val_files_str[:100] + "..."
    print_once(f'\nWord Error Rate: {wer*100.0:5.3f}% on "{val_files_str}"\n')


if __name__ == "__main__":
    logging.configure_logger("RNNT")
    logging.log_start(logging.constants.INIT_START)

    parser = val_arg_parser()
    args = parser.parse_args()
    # check data path args
    if not args.read_from_tar:
        assert (
            args.val_manifests is not None
        ), "Must provide val_manifests if not reading from tar"

    validate(args)
