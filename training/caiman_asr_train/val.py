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

from caiman_asr_train.args.shared import check_shared_args
from caiman_asr_train.args.val import check_val_arguments, val_arg_parser
from caiman_asr_train.evaluate import evaluate
from caiman_asr_train.evaluate.state_resets import check_state_reset_args
from caiman_asr_train.log.tb_dllogger import flush_log
from caiman_asr_train.setup.base import VAL, BuiltObjects
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
    if val_objects.multi_gpu:
        torch.distributed.barrier()

    # A checkpoint should always be specified
    assert args.ckpt is not None

    if val_objects.multi_gpu:
        assert args.nth_batch_only is None, "nth_batch_only not supported in multi-gpu"

    check_state_reset_args(args.sr_segment, args.sr_overlap, args.val_batch_size)

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
    )
    wer = results["wer"]
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
            "/workspace/training/caiman_asr_train/val.py",
        )
        val_objects = ValSetup().run(args)

    main(args, val_objects)
