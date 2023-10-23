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


# valCPU.py is derived from val.py
# rob@myrtle, May 2022

import copy
import os
import random
import time
from argparse import ArgumentParser, Namespace

import numpy as np
import torch

from rnnt_train.common import helpers
from rnnt_train.common.data import features
from rnnt_train.common.data.build_dataloader import build_dali_loader
from rnnt_train.common.data.text import Tokenizer
from rnnt_train.common.data.webdataset import LengthUnknownError
from rnnt_train.common.helpers import (
    Checkpointer,
    num_weights,
    print_once,
    process_evaluation_epoch,
)
from rnnt_train.common.seed import set_seed
from rnnt_train.common.stream_norm import StreamNorm
from rnnt_train.common.tb_dllogger import flush_log, init_log, log
from rnnt_train.rnnt import config
from rnnt_train.rnnt.decoder import RNNTGreedyDecoder
from rnnt_train.rnnt.model import RNNT
from rnnt_train.val import val_arg_parser


def val_cpu_arg_parser() -> ArgumentParser:
    parser = val_arg_parser()
    val_cpu_args = parser.add_argument_group("CPU validation args")

    val_cpu_args.add_argument(
        "--streaming_normalization",
        action="store_true",
        default=False,
        help="Use streaming normalization instead of DALI normalization.",
    )
    val_cpu_args.add_argument(
        "--reset_stream_stats",
        action="store_true",
        default=False,
        help="Reset streaming normalization statistics for every sentence.",
    )
    val_cpu_args.add_argument(
        "--alpha",
        default=0.001,
        type=float,
        help="Streaming normalization decay coefficient, 0<=alpha<1",
    )
    val_cpu_args.add_argument(
        "--dump_nth",
        default=None,
        type=int,
        help="Dump dither-off tensors from the nth batch to /results/ and exit",
    )
    val_cpu_args.add_argument(
        "--dump_preds",
        action="store_true",
        default=False,
        help="Dump text predictions to /results/preds.txt",
    )

    return parser


@torch.no_grad()
def evaluate(
    epoch,
    step,
    val_loader,
    val_feat_proc,
    detokenize,
    ema_model,
    greedy_decoder,
    stream_norm,
    args,
):
    ema_model.eval()

    dumptype = None
    if args.dump_nth != None:
        if stream_norm:
            dumptype = "stream"
        else:
            dumptype = "dali"

    if args.reset_stream_stats:
        training_means = torch.load("/results/melmeans.pt")
        training_vars = torch.load("/results/melvars.pt")

    start_time = time.time()
    agg = {"preds": [], "txts": [], "idx": []}

    try:
        total_loader_len = f"{len(val_loader):<10}"
    except LengthUnknownError:
        total_loader_len = "unk"
    for i, batch in enumerate(val_loader):
        print(
            f"{val_loader.pipeline_type} evaluation: {i:>10}/{total_loader_len}",
            end="\r",
        )

        # note : these variable names are a bit misleading : 'audio' is already features - rob@myrtle
        audio, audio_lens, txt, txt_lens = batch

        if args.dump_nth != None and i == args.dump_nth and stream_norm:
            np.save(f"/results/logmels{i}.npy", audio.numpy())

        if stream_norm:
            # Then the audio tensor was not normalized by DALI and must be normalized here.
            # The Rust Inference Server inits each new Channel with the stream-norm training stats.
            # The args.reset_stream_stats option acts similarly for each new utterance in the manifest.
            # The Python system can then match the Rust system using a new Channel for each utterance.
            if args.reset_stream_stats:
                stream_norm.mel_means = training_means.clone()
                stream_norm.mel_vars = training_vars.clone()
            # The stream_norm class normalizes over time using an exponential moving average.
            # The stats held in the stream_norm class are updated each frame, effectively adapting to
            # the current speaker.  audio is (batch, meldim, time) and batch is enforced to be 1 below.
            for j in range(audio.shape[2]):
                audio[0, :, j] = stream_norm.normalize(audio[0, :, j])

        # By default the stream-norm stats are updated by every utterance in the manifest.
        # This continue statement therefore follows the update (above).
        if args.dump_nth != None and i < args.dump_nth:
            continue

        if args.dump_nth != None:
            np.save(f"/results/{dumptype}norm{i}.npy", audio.numpy())

        # now do frame stacking - rob@myrtle
        feats, feat_lens = val_feat_proc([audio, audio_lens])

        if args.dump_nth != None:
            np.save(f"/results/{dumptype}stack{i}.npy", feats.numpy())

        pred = greedy_decoder.decode(ema_model, feats, feat_lens, dumptype, i)

        agg["preds"] += helpers.gather_predictions([pred], detokenize)
        agg["txts"] += helpers.gather_transcripts(
            [txt.cpu()], [txt_lens.cpu()], detokenize
        )

        if args.dump_nth != None:
            with open(f"/results/{dumptype}preds{i}.txt", "w") as f:
                for line in agg["preds"]:
                    f.write(str(line))
                    f.write("\n")
            with open(f"/results/txts{i}.txt", "w") as f:
                for line in agg["txts"]:
                    f.write(str(line))
                    f.write("\n")
            exit()

    wer, loss = process_evaluation_epoch(agg)

    log(
        (epoch,),
        step,
        "dev_ema",
        {"loss": loss, "wer": 100.0 * wer, "took": time.time() - start_time},
    )

    if args.dump_preds:
        with open(f"/results/preds.txt", "w") as f:
            for line in agg["preds"]:
                f.write(str(line))
                f.write("\n")

    return wer


def validate_cpu(args: Namespace, return_dataloader: bool = False) -> None:
    """
    Validates model on CPU on dataset in args.val_manifests or args.val_tar_files.
    """
    if args.streaming_normalization:
        if args.val_batch_size != 1:
            print("Streaming normalization requires val_batch_size of 1")
            exit()

    if args.dump_nth != None:
        if args.val_batch_size != 1:
            print("dump_nth requires val_batch_size of 1 (to prevent logp overwrites)")
            exit()

    # Set PyTorch to run on one CPU thread to ensure deterministic PyTorch output.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if args.seed is not None:
        set_seed(args.seed, args.local_rank)

    init_log(args)

    cfg = config.load(args.model_config)

    print_once("Setting up datasets...")
    (
        val_dataset_kw,
        val_features_kw,
        val_splicing_kw,
        val_specaugm_kw,
    ) = config.input(cfg, "val")

    if args.dump_nth != None:
        val_features_kw["dither"] = 0.0

    tokenizer_kw = config.tokenizer(cfg)
    tokenizer = Tokenizer(**tokenizer_kw)

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
        cpu=True,
    )

    val_feat_proc = val_augmentations

    # set up the model
    rnnt_config = config.rnnt(cfg)
    rnnt_config["gpu_unavailable"] = True
    model = RNNT(n_classes=tokenizer.num_labels + 1, **rnnt_config)
    blank_idx = tokenizer.num_labels
    greedy_decoder = RNNTGreedyDecoder(
        blank_idx=blank_idx, max_symbol_per_sample=args.max_symbol_per_sample
    )

    print_once(f"Model size: {num_weights(model) / 10**6:.1f}M params\n")

    ema_model = copy.deepcopy(model)

    # A checkpoint should always be specified
    assert args.ckpt is not None

    # setup checkpointer
    checkpointer = Checkpointer(args.output_dir, "RNN-T")

    # load checkpoint (modified to not need optimizer / meta args)
    checkpointer.load(args.ckpt, model, ema_model)

    # setup streaming normalizer
    if args.streaming_normalization:
        melmeans = torch.load("/results/melmeans.pt")
        melvars = torch.load("/results/melvars.pt")
        stream_norm = StreamNorm(args.alpha, melmeans, melvars)
        print_once(
            f"Using streaming normalization, alpha={args.alpha}, "
            f"reset_stream_stats={args.reset_stream_stats}"
        )
    else:
        stream_norm = None

    epoch = 1
    step = None  # Switches off logging of val data results to TensorBoard

    wer = evaluate(
        epoch,
        step,
        val_loader,
        val_feat_proc,
        tokenizer.detokenize,
        ema_model,
        greedy_decoder,
        stream_norm,
        args,
    )

    flush_log()
    val_files = args.val_manifests if not args.read_from_tar else args.val_tar_files
    val_files_str = " ".join(val_files)
    if len(val_files_str) > 100:
        val_files_str = val_files_str[:100] + "..."
    print_once(f'\nWord Error Rate: {wer*100.0:5.3f}% on "{val_files_str}"\n')


if __name__ == "__main__":
    parser = val_cpu_arg_parser()
    args = parser.parse_args()
    # check data path args
    if not args.read_from_tar:
        assert (
            args.val_manifests is not None
        ), "Must provide val_manifests if not reading from tar"

    validate_cpu(args)
