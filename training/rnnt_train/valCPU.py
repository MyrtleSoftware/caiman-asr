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

import time
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from beartype.typing import Any, Dict

from rnnt_train.common import helpers
from rnnt_train.common.data.webdataset import LengthUnknownError
from rnnt_train.common.helpers import print_once, process_evaluation_epoch
from rnnt_train.common.shared_args import check_shared_args
from rnnt_train.common.stream_norm import StreamNorm
from rnnt_train.common.tb_dllogger import flush_log, log
from rnnt_train.setup.base import VAL, BuiltObjects
from rnnt_train.setup.val_cpu import ValCPUSetup
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
    decoder,
    stream_norm,
    args,
) -> Dict[str, Any]:
    """
    Perform on-CPU evaluation.
    """
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
    results = {"preds": [], "txts": [], "idx": []}

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

        pred = decoder.decode(ema_model, feats, feat_lens, dumptype, i)

        results["preds"] += helpers.gather_predictions([pred], detokenize)
        results["txts"] += helpers.gather_transcripts(
            [txt.cpu()], [txt_lens.cpu()], detokenize
        )

        if args.dump_nth != None:
            with open(f"/results/{dumptype}preds{i}.txt", "w") as f:
                for line in results["preds"]:
                    f.write(str(line))
                    f.write("\n")
            with open(f"/results/txts{i}.txt", "w") as f:
                for line in results["txts"]:
                    f.write(str(line))
                    f.write("\n")
            exit()

    wer, loss = process_evaluation_epoch(results)
    results["wer"] = wer

    log(
        (epoch,),
        step,
        "dev_ema",
        {"loss": loss, "wer": 100.0 * wer, "took": time.time() - start_time},
    )

    if args.dump_preds:
        with open(f"{args.output_dir}/preds.txt", "w") as f:
            for line in results["preds"]:
                f.write(str(line))
                f.write("\n")

    return results


def validate_cpu(
    args: Namespace, val_objects: BuiltObjects, return_dataloader: bool = False
) -> Dict[str, Any]:
    """
    Validates model on CPU on dataset in args.val_manifests or args.val_tar_files.

    If return_dataloader=True, return dataloader in results dict.
    """
    if args.streaming_normalization:
        if args.val_batch_size != 1:
            print("Streaming normalization requires val_batch_size of 1")
            exit()

    if args.dump_nth != None:
        if args.val_batch_size != 1:
            print("dump_nth requires val_batch_size of 1 (to prevent logp overwrites)")
            exit()

    # A checkpoint should always be specified
    assert args.ckpt is not None

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

    val_loader = val_objects.data_objects[VAL].loader
    results = evaluate(
        epoch,
        step,
        val_loader,
        val_objects.feat_procs[VAL],
        val_objects.tokenizers[VAL].detokenize,
        val_objects.ema_model,
        val_objects.decoder,
        stream_norm,
        args,
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

    validate_cpu(args, val_objects)


if __name__ == "__main__":
    parser = val_cpu_arg_parser()
    args = parser.parse_args()
    val_objects = ValCPUSetup().run(args)
    main(args, val_objects)
