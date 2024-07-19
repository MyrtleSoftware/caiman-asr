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

from caiman_asr_train.data.webdataset import LengthUnknownError
from caiman_asr_train.evaluate.distributed import process_evaluation_epoch
from caiman_asr_train.evaluate.metrics import word_error_rate
from caiman_asr_train.evaluate.state_resets import (
    state_resets_merge_segments,
    state_resets_reshape_feats,
)
from caiman_asr_train.latency.timestamp import group_timestamps
from caiman_asr_train.log.tb_dllogger import log
from caiman_asr_train.rnnt.model_forward import model_loss_forward_val
from caiman_asr_train.train_utils.distributed import get_rank_or_zero


def __rnnt_decoder_predictions_tensor(tensor, detokenize):
    """
    Takes output of greedy rnnt decoder and converts to strings.
    Args:
        tensor: model output tensor
        label: A list of labels
    Returns:
        prediction
    """
    return [detokenize(pred) for pred in tensor]


def gather_losses(losses_list):
    return [torch.mean(torch.stack(losses_list))]


def gather_predictions(predictions_list, detokenize):
    rnnt_predictions = (
        __rnnt_decoder_predictions_tensor(prediction, detokenize)
        for prediction in predictions_list
    )

    return [prediction for batch in rnnt_predictions for prediction in batch]


def gather_transcripts(transcript_list, transcript_len_list, detokenize):
    return [
        detokenize(t[:l].long().cpu().numpy().tolist())
        for txt, lens in zip(transcript_list, transcript_len_list)
        for t, l in zip(txt, lens)
    ]


def calculate_wer(preds, tgt, tgt_lens, detokenize, standardize_wer):
    """
    Calculates WER and returns an example hypothesis and reference.
    """
    with torch.no_grad():
        references = gather_transcripts([tgt], [tgt_lens], detokenize)
        hypotheses = __rnnt_decoder_predictions_tensor(preds, detokenize)

    wer, _, _ = word_error_rate(hypotheses, references, standardize=standardize_wer)
    return wer, hypotheses[0], references[0]


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
    standardize_wer=True,
    calculate_loss=False,
    nth_batch_only=None,
    using_cpu=False,
) -> Dict[str, Any]:
    """
    Perform evaluation.
    """
    ema_model.eval()
    enc_time_reduction = ema_model.enc_stack_time_factor

    start_time = time.time()
    results = {"preds": [], "txts": [], "idx": [], "timestamps": [], "token_probs": []}
    if calculate_loss:
        results["losses"] = []

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
        audio, audio_lens, txt, txt_lens, raw_transcripts = batch

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
            results["losses"] += gather_losses([loss.cpu()])

        # if state resets is on, prepare the feats
        if args.sr_segment:
            (
                feats,
                feat_lens,
                sr_segment_frames,
                sr_overlap_frames,
            ) = state_resets_reshape_feats(
                args.sr_segment, args.sr_overlap, cfg, feats, feat_lens
            )

        pred, timestamps, probs = decoder.decode(feats, feat_lens)

        # if state resets is used, merge predictions and timestamps of segments into one
        if args.sr_segment:
            pred, timestamps, probs = state_resets_merge_segments(
                pred,
                timestamps,
                probs,
                enc_time_reduction,
                sr_segment_frames,
                sr_overlap_frames,
            )

        # For each predicted sentence, detokenize token into subword
        subwords = [[detokenize(token) for token in sentence] for sentence in pred]
        if probs:
            token_probs = [
                [
                    (token, round(prob, 4))
                    for token, prob in zip(token_list, prob_list, strict=True)
                ]
                for token_list, prob_list in zip(subwords, probs, strict=True)
            ]
            results["token_probs"] += token_probs

        preds = gather_predictions([pred], detokenize)
        results["preds"] += preds
        results["txts"] += raw_transcripts
        if timestamps:
            # convert token timestamps to word timestamps
            word_timestamps = group_timestamps(subwords, timestamps, preds)
            results["timestamps"] += word_timestamps

    wer, loss = process_evaluation_epoch(
        results,
        standardize_wer=standardize_wer,
        breakdown_wer=args.breakdown_wer,
        breakdown_chars=args.breakdown_chars,
    )
    results["wer"] = wer

    log(
        (epoch,),
        step,
        "dev_ema",
        {"loss": loss, "wer": 100.0 * wer, "took": time.time() - start_time},
    )

    if args.dump_preds:
        with open(f"{args.output_dir}/preds{get_rank_or_zero()}.txt", "w") as f:
            for line in results["preds"]:
                f.write(str(line))
                f.write("\n")

    ema_model.train()
    return results
