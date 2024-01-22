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

# The evaluate() function was originally in train.py - rob@myrtle

import time

import torch
from beartype.typing import Any, Dict

from rnnt_train.common import helpers
from rnnt_train.common.data.webdataset import LengthUnknownError
from rnnt_train.common.helpers import process_evaluation_epoch
from rnnt_train.common.tb_dllogger import log


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
    args,
    calculate_loss=True,
) -> Dict[str, Any]:
    """
    Perform on-GPU evaluation.
    """
    ema_model.eval()
    enc_time_reduction = ema_model.enc_stack_time_factor

    start_time = time.time()
    if calculate_loss:
        results = {"losses": [], "preds": [], "txts": [], "idx": []}
    else:
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
        # txt is      (batch, max(txt_lens))
        # txt_lens is (batch, )
        audio, audio_lens, txt, txt_lens = batch

        # if these tensors were computed on cpu then move them to gpu - rob@myrtle
        if args.dali_device == "cpu":
            audio = audio.cuda()
            audio_lens = audio_lens.cuda()
            txt = txt.cuda()
            txt_lens = txt_lens.cuda()

        # now do frame stacking - rob@myrtle
        feats, feat_lens = val_feat_proc([audio, audio_lens])

        # batch_offset and max_f_len parameters are required for the apex transducer
        # loss/joint implementations
        final_feat_lens = (feat_lens + enc_time_reduction - 1) // enc_time_reduction
        batch_offset = torch.cumsum(final_feat_lens * (txt_lens + 1), dim=0)
        max_f_len = max(final_feat_lens)

        if calculate_loss:
            # note : more misleading variable names : 'log_prob*' are actually logits - rob@myrtle
            log_probs, log_prob_lens, _ = ema_model(
                feats, feat_lens, txt, txt_lens, batch_offset=batch_offset
            )
            max_f_len = max(final_feat_lens)
            loss = loss_fn(
                log_probs,
                log_prob_lens,
                txt,
                txt_lens,
                batch_offset,
                max_f_len,
            )

        pred = decoder.decode(ema_model, feats, feat_lens)

        if calculate_loss:
            results["losses"] += helpers.gather_losses([loss.cpu()])
        results["preds"] += helpers.gather_predictions([pred], detokenize)
        results["txts"] += helpers.gather_transcripts(
            [txt.cpu()], [txt_lens.cpu()], detokenize
        )

    wer, loss = process_evaluation_epoch(results)
    results["wer"] = wer

    log(
        (epoch,),
        step,
        "dev_ema",
        {"loss": loss, "wer": 100.0 * wer, "took": time.time() - start_time},
    )
    ema_model.train()
    return results
