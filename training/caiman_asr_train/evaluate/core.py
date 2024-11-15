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


import json
import time

import torch
from beartype import beartype
from beartype.typing import Any, Dict

from caiman_asr_train.data.webdataset import LengthUnknownError
from caiman_asr_train.evaluate.distributed import process_evaluation_epoch
from caiman_asr_train.evaluate.error_rates import ErrorRate, error_rate_abbrev
from caiman_asr_train.evaluate.metrics import word_error_rate
from caiman_asr_train.evaluate.state_resets import (
    state_resets_merge_batched_segments,
    state_resets_reshape_batched_feats,
)
from caiman_asr_train.evaluate.state_resets.timestamp import (
    FullStamp,
    user_perceived_time,
)
from caiman_asr_train.evaluate.trim import EOSTrimConfig, trim_predictions
from caiman_asr_train.latency.ctm import get_reference_ctms, manage_ctm_export
from caiman_asr_train.latency.timestamp import (
    EOS,
    SequenceTimestamp,
    Silence,
    group_timestamps,
)
from caiman_asr_train.log.tb_dllogger import log
from caiman_asr_train.rnnt import config
from caiman_asr_train.rnnt.model_forward import model_loss_forward_val
from caiman_asr_train.train_utils.distributed import get_rank_or_zero
from caiman_asr_train.utils.frame_width import (
    encoder_output_frame_width,
    input_feat_frame_width,
)
from caiman_asr_train.utils.responses import fuse_partials, split_batched_finals


@beartype
def count_eos(seq: SequenceTimestamp) -> int:
    """
    Returns 1 if the sequence ends with an EOS token, 0 otherwise.
    """
    match seq.eos:
        case EOS(_):
            return 1
        case _:
            return 0


@beartype
def count_sil(seq: SequenceTimestamp) -> int:
    """
    Returns 1 if the sequence ends with a Silence token, 0 otherwise.
    """
    match seq.eos:
        case Silence(_):
            return 1
        case _:
            return 0


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


def calculate_wer(preds, tgt, tgt_lens, detokenize, standardize_wer, error_rate):
    """
    Calculates WER and returns an example hypothesis and reference.
    """
    with torch.no_grad():
        references = gather_transcripts([tgt], [tgt_lens], detokenize)
        hypotheses = __rnnt_decoder_predictions_tensor(preds, detokenize)

    wer, _, _ = word_error_rate(
        hypotheses, references, error_rate, standardize=standardize_wer
    )
    return wer, hypotheses[0], references[0]


@beartype
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
    error_rate: ErrorRate,
    standardize_wer=True,
    calculate_loss=False,
    nth_batch_only=None,
    using_cpu=False,
    skip_logging=False,
) -> Dict[str, Any]:
    """
    Perform evaluation.
    """
    ema_model.eval()
    enc_time_reduction = ema_model.enc_stack_time_factor

    start_time = time.time()

    # Potentially get reference CTMs for emission latency evaluation
    if args.calculate_emission_latency:
        reference_ctms = get_reference_ctms(args)

    results = {
        "preds": [],
        "txts": [],
        "idx": [],
        "timestamps": [],
        "token_probs": [],
        "fnames": [],
    }
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
        audio, audio_lens, txt, txt_lens, raw_transcripts, fnames = batch

        # move tensors back to gpu, unless cpu is used during validation
        if args.dali_val_device == "cpu" and not using_cpu:
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
            (feats, feat_lens, meta) = state_resets_reshape_batched_feats(
                args.sr_segment, args.sr_overlap, cfg, feats, feat_lens
            )

        responses = decoder.decode(feats, feat_lens)

        pred, model_t, probs = split_batched_finals(responses)

        responses = list(map(fuse_partials, responses))

        _, emit_t, _ = split_batched_finals(responses)

        timestamps = list(
            list(map(lambda y: FullStamp(*y), zip(*x))) for x in zip(model_t, emit_t)
        )

        # if state resets is used, merge predictions and timestamps of segments into one
        if args.sr_segment:
            pred, timestamps, probs = state_resets_merge_batched_segments(
                pred,
                timestamps,
                probs,
                enc_time_reduction,
                meta,
                decoder.eos_index if args.eos_is_terminal else None,
            )

        # Strip model timestamp
        timestamps = [[user_perceived_time(t) for t in ts] for ts in timestamps]

        i_width = input_feat_frame_width(config.load(args.model_config))
        o_width = encoder_output_frame_width(args.model_config)

        if (eos_idx := decoder.eos_index) is None:
            eos_info = None
        else:
            eos_info = EOSTrimConfig(
                eos_idx=eos_idx,
                eos_is_terminal=args.eos_is_terminal,
                blank_idx=decoder.blank_idx,
            )

        pred, timestamps, probs, last_emit_time = trim_predictions(
            pred,
            timestamps,
            probs,
            i_width,
            o_width,
            feat_lens,
            eos_vad_threshold=args.eos_vad_threshold,
            eos_info=eos_info,
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
        results["fnames"] += fnames
        results["txts"] += raw_transcripts

        if timestamps:
            # convert token timestamps to word timestamps
            word_timestamps = group_timestamps(
                subwords, timestamps, preds, last_emit_time
            )
            results["timestamps"] += word_timestamps

    end_time = time.time()

    wer, loss = process_evaluation_epoch(
        results,
        standardize_wer=standardize_wer,
        breakdown_wer=args.breakdown_wer,
        breakdown_chars=args.breakdown_chars,
        error_rate=error_rate,
    )

    abbrev = error_rate_abbrev(error_rate)

    results[abbrev] = wer

    data = {"loss": loss, abbrev: 100.0 * wer, "took": end_time - start_time}

    if "timestamps" in results:
        eos_count = sum(count_eos(seq) for seq in results["timestamps"])
        sil_count = sum(count_sil(seq) for seq in results["timestamps"])
        tot_count = len(results["timestamps"])

        results["eos_frac"] = eos_count / tot_count
        results["sil_frac"] = sil_count / tot_count
        results["rem_frac"] = 1 - results["eos_frac"] - results["sil_frac"]

        data["eos_frac"] = results["eos_frac"]
        data["sil_frac"] = results["sil_frac"]
        data["rem_frac"] = results["rem_frac"]

    if args.calculate_emission_latency:
        if args.read_from_tar:
            flist = None
        else:
            with open(val_loader.sampler.get_file_list_path(), "r") as fh:
                flist = [line.strip().split(" ")[0] for line in fh.readlines()]

        latency_metrics, latencies, sil_latency, eos_latency = manage_ctm_export(
            args,
            results["timestamps"],
            reference_ctms,
            flist,
        )
        results["latency_metrics"] = latency_metrics
        data.update(latency_metrics)

        json_results = {
            "sil_frac": results["sil_frac"],
            "eos_frac": results["eos_frac"],
            "sil_latency": sil_latency,
            "eos_latency": eos_latency,
            "latencies": latencies,
        }

        with open(
            f"{args.output_dir}/latencies{get_rank_or_zero()}_{args.timestamp}_{step}.json",
            "w",
        ) as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

    if not skip_logging:
        log((epoch,), step, "dev_ema", data)

    json_results = [
        {
            "hyp": hyp,
            "ref": ref,
            error_rate_abbrev(error_rate): word_error_rate(
                [hyp], [ref], error_rate, standardize_wer
            )[0],
            "fname": fname,
        }
        for hyp, ref, fname in zip(
            results["preds"], results["txts"], results["fnames"], strict=True
        )
    ]

    with open(
        f"{args.output_dir}/preds{get_rank_or_zero()}_{args.timestamp}_{step}.json", "w"
    ) as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    ema_model.train()
    return results
