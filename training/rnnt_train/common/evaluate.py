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


import time

import torch
from beartype.typing import Any, Dict

from rnnt_train.common import helpers
from rnnt_train.common.data.webdataset import LengthUnknownError
from rnnt_train.common.helpers import is_rank_zero, process_evaluation_epoch
from rnnt_train.common.tb_dllogger import log
from rnnt_train.rnnt.model_forward import model_loss_forward_val
from rnnt_train.utils.timestamp import group_timestamps


@torch.no_grad()
def evaluate(
    epoch,
    step,
    val_loader,
    val_feat_proc,
    detokenize,
    ema_model,
    loss_fn,
    decoder,
    cfg,
    args,
    stream_norm=None,
    standardize_wer=True,
    calculate_loss=False,
    nth_batch_only=None,
    using_cpu=False,
) -> Dict[str, Any]:
    """
    Perform evaluation.
    """
    ema_model.eval()
    # apply streaming normalization
    if stream_norm and not args.dont_reset_stream_stats:
        training_means = torch.load("/results/melmeans.pt")
        training_vars = torch.load("/results/melvars.pt")

    start_time = time.time()
    if calculate_loss:
        results = {"losses": [], "preds": [], "txts": [], "idx": [], "timestamps": []}
    else:
        results = {"preds": [], "txts": [], "idx": [], "timestamps": []}

    try:
        total_loader_len = f"{len(val_loader):<10}"
    except LengthUnknownError:
        total_loader_len = "unk"

    for i, batch in enumerate(val_loader):
        if nth_batch_only is not None:
            if i < nth_batch_only:
                continue
            elif i > nth_batch_only:
                break
        print(
            f"{val_loader.pipeline_type} evaluation: {i:>10}/{total_loader_len}",
            end="\r",
        )

        # note : these variable names are a bit misleading : 'audio' is already features
        # txt is      (batch, max(txt_lens))
        # txt_lens is (batch, )
        audio, audio_lens, txt, txt_lens = batch

        if stream_norm:
            # Then the audio tensor was not normalized by DALI and must be normalized here.
            # The Rust Inference Server inits each new Channel with the stream-norm
            # training stats.
            # The Python default is to act similarly for each new utterance in the
            # manifest.
            # The Python system can then match the Rust system using a new Channel for
            # each utterance.
            if not args.dont_reset_stream_stats:
                stream_norm.mel_means = training_means.clone()
                stream_norm.mel_vars = training_vars.clone()
            # The stream_norm class normalizes over time using an exponential moving
            # average.
            # The stats held in the stream_norm class are updated each frame,
            # effectively adapting to
            # the current speaker. audio is (batch, meldim, time) and batch is enforced
            # to be 1 below.
            for j in range(audio.shape[2]):
                audio[0, :, j] = stream_norm.normalize(audio[0, :, j])

        # move tensors back to gpu, unless cpu is used during validation
        if args.dali_device == "cpu" and not using_cpu:
            audio = audio.cuda()
            audio_lens = audio_lens.cuda()
            txt = txt.cuda()
            txt_lens = txt_lens.cuda()

        # now do frame stacking
        feats, feat_lens = val_feat_proc([audio, audio_lens])

        if calculate_loss:
            loss = model_loss_forward_val(
                ema_model, loss_fn, feats, feat_lens, txt, txt_lens
            )
            results["losses"] += helpers.gather_losses([loss.cpu()])
        pred, timestamps = decoder.decode(ema_model, feats, feat_lens)

        preds = helpers.gather_predictions([pred], detokenize)
        results["preds"] += preds
        results["txts"] += helpers.gather_transcripts(
            [txt.cpu()], [txt_lens.cpu()], detokenize
        )
        if timestamps:
            # For each predicted sentence, detokenize token into subword
            subwords = [[detokenize(token) for token in sentence] for sentence in pred]
            # convert token timestamps to word timestamps
            word_timestamps = group_timestamps(subwords, timestamps, preds)
            results["timestamps"] += word_timestamps

    wer, loss = process_evaluation_epoch(results, standardize_wer=standardize_wer)
    results["wer"] = wer

    log(
        (epoch,),
        step,
        "dev_ema",
        {"loss": loss, "wer": 100.0 * wer, "took": time.time() - start_time},
    )

    if args.dump_preds and is_rank_zero():
        with open(f"{args.output_dir}/preds.txt", "w") as f:
            for line in results["preds"]:
                f.write(str(line))
                f.write("\n")

    ema_model.train()
    return results
