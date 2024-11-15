#! /usr/bin/env python3
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

import os
from argparse import Namespace
from glob import glob

import torch
from beartype.typing import Any, Dict

from caiman_asr_train.args.shared import check_shared_args
from caiman_asr_train.args.val import check_val_arguments, val_arg_parser
from caiman_asr_train.evaluate.core import evaluate
from caiman_asr_train.evaluate.error_rates import error_rate_abbrev, error_rate_long
from caiman_asr_train.log.profiling import finish_profiling, set_up_profiling
from caiman_asr_train.log.tb_dllogger import flush_log
from caiman_asr_train.setup.base import BuiltObjects
from caiman_asr_train.setup.core import VAL
from caiman_asr_train.setup.val import ValCPUSetup, ValSetup
from caiman_asr_train.train_utils.distributed import print_once
from caiman_asr_train.train_utils.torchrun import maybe_restart_with_torchrun


def validate(
    args: Namespace, val_objects: BuiltObjects, return_dataloader: bool = False
) -> Dict[str, Any]:
    """
    Validates model on dataset in args.val_manifests or args.val_tar_files.

    If return_dataloader=True, return dataloader in results dict.
    """
    # A checkpoint should always be specified
    assert args.ckpt is not None

    if val_objects.multi_gpu:
        torch.distributed.barrier()
        assert args.nth_batch_only is None, "nth_batch_only not supported in multi-gpu"

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
        val_objects.cfg,
        args=args,
        standardize_wer=val_objects.data_objects[VAL].dataset_kw["standardize_wer"],
        calculate_loss=args.calculate_loss,
        nth_batch_only=args.nth_batch_only,
        using_cpu=args.cpu,
        error_rate=val_objects.error_rate,
    )
    wer = results[error_rate_abbrev(val_objects.error_rate)]
    flush_log()
    if not args.val_from_dir:
        if args.read_from_tar:
            val_files = args.val_tar_files
        elif args.use_hugging_face:
            val_files = [
                f"dataset={args.hugging_face_val_dataset},",
                f"config={args.hugging_face_val_config},",
                f"split={args.hugging_face_val_split},",
                f"transcript_key={args.hugging_face_val_transcript_key}",
            ]
        else:
            val_files = args.val_manifests
        val_files_str = " ".join(val_files)
        if len(val_files_str) > 200:
            val_files_str = val_files_str[:200] + "..."
        print_once(
            f"\n{error_rate_long(val_objects.error_rate)}: "
            f'{wer*100.0:5.3f}% on "{val_files_str}"\n'
        )
    else:
        print_once(
            f"\n{error_rate_long(val_objects.error_rate)}: "
            f"{wer*100.0:5.3f}% on {args.val_audio_dir=} and "
            f"{args.val_txt_dir=}\n"
        )

    if return_dataloader:
        results["dataloader"] = val_loader

    return results


def build_objects(args):
    profilers = set_up_profiling(args.profiler, args.output_dir, args.timestamp)
    val_objects = ValCPUSetup().run(args) if args.cpu else ValSetup().run(args)
    return val_objects, profilers


def validation_routine(args, build_objects_func, validation_func):
    check_shared_args(args)
    check_val_arguments(args)
    if os.path.isdir(args.ckpt):
        # If directory, iterate over all ckpts in given dir
        ckpt_directory = args.ckpt
        ckpt_files = glob(os.path.join(ckpt_directory, "*.pt"))
        if not ckpt_files:
            raise ValueError(f"No checkpoint files found in {args.ckpt}")
        wers = {}
        for i, ckpt_file in enumerate(ckpt_files):
            if i > 0:
                args.skip_init = True
            args.ckpt = ckpt_file
            if args.local_rank == 0:
                print(f"{i+1}/{len(ckpt_files)} Validating checkpoint: {ckpt_file}")
            val_objects, profilers = build_objects_func(args)
            results = validation_func(args, val_objects)
            wers[ckpt_file] = round(
                results[error_rate_abbrev(val_objects.error_rate)] * 100, 3
            )
            finish_profiling(args.profiler, args.output_dir, profilers, args.timestamp)
        print_once(f"{error_rate_abbrev(val_objects.error_rate).upper()}s: {wers}")

    else:
        if args.local_rank == 0:
            print(f"Validating checkpoint: {args.ckpt}")
        val_objects, profilers = build_objects_func(args)
        validation_func(args, val_objects)
        finish_profiling(args.profiler, args.output_dir, profilers, args.timestamp)


def run_validate(args: Namespace, val_objects: BuiltObjects):
    # check data path args
    if not args.read_from_tar:
        assert (
            args.val_manifests is not None
        ), "Must provide val_manifests if not reading from tar"
    return validate(args, val_objects=val_objects)


if __name__ == "__main__":
    parser = val_arg_parser()
    args = parser.parse_args()
    if not args.cpu:
        maybe_restart_with_torchrun(
            args.num_gpus,
            args.called_by_torchrun,
            "/workspace/training/caiman_asr_train/val.py",
        )
    validation_routine(args, build_objects, run_validate)
