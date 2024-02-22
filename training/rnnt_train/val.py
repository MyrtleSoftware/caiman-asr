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

from argparse import Namespace

import torch
from beartype.typing import Any, Dict

from rnnt_train.args.val import check_val_arguments, val_arg_parser
from rnnt_train.common.evaluate import evaluate
from rnnt_train.common.helpers import print_once
from rnnt_train.common.shared_args import check_shared_args
from rnnt_train.common.stream_norm import StreamNorm
from rnnt_train.common.tb_dllogger import flush_log
from rnnt_train.common.tee import start_logging_stdout_and_stderr
from rnnt_train.common.torchrun import maybe_restart_with_torchrun
from rnnt_train.setup.base import VAL, BuiltObjects
from rnnt_train.setup.val import ValCPUSetup, ValSetup


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

    # setup streaming normalizer
    if args.streaming_normalization:
        melmeans = torch.load("/results/melmeans.pt")
        melvars = torch.load("/results/melvars.pt")
        stream_norm = StreamNorm(args.alpha, melmeans, melvars)
        print_once(
            f"Using streaming normalization, alpha={args.alpha}, "
            f"reset_stream_stats={not args.dont_reset_stream_stats}"
        )
    else:
        stream_norm = None

    if val_objects.multi_gpu:
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
        stream_norm=stream_norm,
        args=args,
        standardize_wer=val_objects.data_objects[VAL].dataset_kw["standardize_wer"],
        calculate_loss=args.calculate_loss,
        nth_batch_only=args.nth_batch_only,
        using_cpu=args.cpu,
    )
    wer = results["wer"]
    flush_log()
    if not args.val_from_dir:
        val_files = args.val_manifests if not args.read_from_tar else args.val_tar_files
        val_files_str = " ".join(val_files)
        if len(val_files_str) > 100:
            val_files_str = val_files_str[:100] + "..."
        print_once(f'\nWord Error Rate: {wer*100.0:5.3f}% on "{val_files_str}"\n')
    else:
        print_once(
            f"\nWord Error Rate: {wer*100.0:5.3f}% on {args.val_audio_dir=} and "
            f"{args.val_txt_dir=}\n"
        )

    if return_dataloader:
        results["dataloader"] = val_loader

    return results


def main(args, val_objects):
    check_shared_args(args)
    check_val_arguments(args)
    # check data path args
    if not args.read_from_tar:
        assert (
            args.val_manifests is not None
        ), "Must provide val_manifests if not reading from tar"
    return validate(args, val_objects=val_objects)


if __name__ == "__main__":
    parser = val_arg_parser()
    args = parser.parse_args()
    if args.cpu:
        val_objects = ValCPUSetup().run(args)
    else:
        maybe_restart_with_torchrun(
            args.num_gpus,
            args.called_by_torchrun,
            "/workspace/training/rnnt_train/val.py",
        )
        val_objects = ValSetup().run(args)

    start_logging_stdout_and_stderr(args.output_dir, args.timestamp, "validation")
    main(args, val_objects)
