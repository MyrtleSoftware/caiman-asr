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
from argparse import Namespace
from pathlib import Path

import torch
from beartype import beartype
from beartype.typing import Any, Dict
from lhotse import Recording
from lhotse.cut import Cut, MixedCut

from caiman_asr_train.core.batches import TimeFirst
from caiman_asr_train.evaluate.distributed import process_evaluation_epoch
from caiman_asr_train.evaluate.error_rates import ErrorRate, error_rate_abbrev
from caiman_asr_train.evaluate.metrics import (
    PerDetails,
    WerDetails,
    compute_keyword_metrics,
    punctuation_error_rate,
    word_error_rate,
)
from caiman_asr_train.evaluate.state_resets import (
    state_resets_merge_batched_segments,
    state_resets_reshape_batched_feats,
)
from caiman_asr_train.evaluate.state_resets.timestamp import (
    FullStamp,
    final_time,
    model_time,
    user_perceived_time,
)
from caiman_asr_train.evaluate.trim import EOSTrimConfig, trim_predictions
from caiman_asr_train.keywords.process import return_list_of_keywords
from caiman_asr_train.latency.ctm import get_reference_manifest_files, manage_ctm_export
from caiman_asr_train.latency.timestamp import (
    EOS,
    SequenceTimestamp,
    Silence,
    frame_to_time,
    group_timestamps,
)
from caiman_asr_train.lhotse.pipeline import LhotsePipeline
from caiman_asr_train.log.tb_dllogger import log
from caiman_asr_train.rnnt import config
from caiman_asr_train.rnnt.joint_loss import JointLoss
from caiman_asr_train.rnnt.loss import IDENTITY_LOSS_MODIFIERS, LossModifiers
from caiman_asr_train.rnnt.model_forward import val_loss
from caiman_asr_train.train_utils.distributed import get_rank_or_zero
from caiman_asr_train.utils.frame_width import (
    encoder_output_frame_width,
    initial_padding,
    input_feat_frame_width,
)
from caiman_asr_train.utils.iter import lmap, lstarmap_zip
from caiman_asr_train.utils.prefix import broadcast_prefix
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


def gather_transcripts(transcript_list, transcript_len_list, detokenize) -> list:
    return [
        detokenize(t[:l].long().cpu().numpy().tolist())
        for txt, lens in zip(transcript_list, transcript_len_list)
        for t, l in zip(txt, lens)
    ]


def calculate_wer(preds, tgt, tgt_lens, detokenize, lang_code, error_rate):
    """
    Calculates WER and returns an example hypothesis and reference.
    """
    with torch.no_grad():
        references = gather_transcripts([tgt], [tgt_lens], detokenize)
        hypotheses = __rnnt_decoder_predictions_tensor(preds, detokenize)

    wer = word_error_rate(hypotheses, references, error_rate, lang_code).wer
    return wer, hypotheses[0], references[0]


@beartype
def calculate_per(
    preds, tgt, tgt_lens, detokenize, punctuation_marks, punctuation_normalization
) -> float:
    """
    Calculates PER

    Parameters
    ----------
    preds
        model predictions tensor
    tgt
        target tensor
    tgt_lens
        target lengths
    detokenize
        detokenization function
    punctuation_marks
        punctuation marks that the PER will be calculated on
    punctuation_normalization
        punctuation normalization rules for the PER calculation

    Returns
    -------
    per: Punctuation Error Rate

    """
    with torch.no_grad():
        references = gather_transcripts([tgt], [tgt_lens], detokenize)
        hypotheses = __rnnt_decoder_predictions_tensor(preds, detokenize)

    per_details = punctuation_error_rate(
        hypotheses, references, punctuation_marks, punctuation_normalization
    )
    return per_details.per


@beartype
def _infer_audio_path(cut: Cut) -> str:
    """
    Lhotse doesn't really have the concept of an audio file path because cuts are
    much more general than that (they can be multi-recording, multi-source, etc).
    However, we need to recover the file name used as the CTM key hence, we make
    this best-effort attempt to do so.
    """

    if isinstance(cut, MixedCut) and cut.first_non_padding_cut.has_custom(
        "original_source"
    ):
        # Use the custom field if available
        sources = cut.first_non_padding_cut.custom["original_source"]

        assert (
            len(sources) == 1
        ), f"Expected 1 original source, found {len(sources)} in cut: {cut}"

        return sources[0]

    # Fallback to reading from the recording source

    recs = [manifest for key, manifest in cut.iter_data() if key == "recording"]
    assert len(recs) == 1, f"Expected 1 recording, found {len(recs)} in cut: {cut}"
    rec = recs[0]
    assert isinstance(rec, Recording)

    assert (
        len(rec.sources) == 1
    ), f"Expected 1 source, found {len(rec.sources)} in cut: {cut}"

    src = rec.sources[0]
    file_name = src.source
    assert isinstance(
        file_name, str
    ), f"Expected source to be a string, found {type(file_name)}"
    assert file_name, f"Cannot infer audio file name from cut: {recs}"

    return file_name


def _strip_data_dir(f: str, data_dir: str) -> str:
    head = Path(data_dir).resolve()
    full = Path(f).resolve()

    try:
        return str(full.relative_to(head))
    except ValueError:
        return f


def _infer_transcript(cut: Cut):
    sup = cut.supervisions
    assert len(sup) == 1, f"Expected 1 supervision, found {len(sup)}"
    return sup[0].text


@beartype
def _dump_preds_to_json(
    all_wer_details: list[WerDetails],
    all_per_details: list[PerDetails],
    preds: list[str],
    refs: list[str],
    fnames: list[str],
    error_rate: ErrorRate,
    calculate_per: bool,
    output_path: str,
) -> None:
    """Populate a JSON file with per-utterance prediction metrics."""
    abbrev = error_rate_abbrev(error_rate)
    json_results = []
    for hyp, ref, fname, wer_details, per_details in zip(
        preds, refs, fnames, all_wer_details, all_per_details, strict=True
    ):
        tot_err = wer_details.scores
        insertions = wer_details.insertions
        deletions = wer_details.deletions
        substitutions = wer_details.substitutions
        per_value = per_details.per if calculate_per else -1.0
        json_results.append(
            {
                "hyp": hyp,
                "ref": ref,
                "norm_hyp": wer_details.norm_hyp,
                "norm_ref": wer_details.norm_ref,
                abbrev: wer_details.wer,
                r"%ins": insertions / tot_err * 100 if tot_err > 0 else 0,
                r"%del": deletions / tot_err * 100 if tot_err > 0 else 0,
                r"%sub": substitutions / tot_err * 100 if tot_err > 0 else 0,
                "per": per_value,
                "fname": fname,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)


@beartype
def _add_wer_metrics(
    results: dict,
    data: dict,
    all_wer_details: list[WerDetails],
    error_rate: ErrorRate,
) -> None:
    """Populate results/data with aggregated WER"""
    avg_wer_details = WerDetails.aggregate(all_wer_details)
    tot_err = avg_wer_details.scores
    abbrev = error_rate_abbrev(error_rate)
    data[abbrev] = 100.0 * avg_wer_details.wer
    data[r"%ins"] = avg_wer_details.insertions / tot_err * 100 if tot_err > 0 else 0
    data[r"%del"] = avg_wer_details.deletions / tot_err * 100 if tot_err > 0 else 0
    data[r"%sub"] = avg_wer_details.substitutions / tot_err * 100 if tot_err > 0 else 0
    results[abbrev] = avg_wer_details.wer


@beartype
def _add_per_metrics(
    results: dict,
    data: dict,
    all_per_details: list[PerDetails],
) -> None:
    """Populate results/data with aggregated PER"""
    avg_per_details = PerDetails.aggregate(all_per_details)
    data["per"] = 100.0 * avg_per_details.per
    results["per"] = avg_per_details.per


@beartype
def _add_keyword_metrics(
    results: dict,
    data: dict,
    args,
) -> None:
    """Populate results/data with keyword metrics."""
    keywords = return_list_of_keywords(args.keywords_path)
    keyword_metrics = compute_keyword_metrics(
        results["preds"],
        results["txts"],
        keywords,
        args.val_standardizer_lang,
    )
    results["keyword_metrics"] = keyword_metrics
    data.update(keyword_metrics)


@beartype
def _add_timestamp_fractions(results: dict, data: dict) -> None:
    """Populate results/data with EOS, Silence, and remaining fractions."""
    timestamps = results["user_perceived_timestamps"]
    tot_count = len(timestamps)
    eos_count = sum(count_eos(seq) for seq in timestamps)
    sil_count = sum(count_sil(seq) for seq in timestamps)
    results["eos_frac"] = eos_count / tot_count
    results["sil_frac"] = sil_count / tot_count
    results["rem_frac"] = 1 - results["eos_frac"] - results["sil_frac"]

    data["eos_frac"] = results["eos_frac"]
    data["sil_frac"] = results["sil_frac"]
    data["rem_frac"] = results["rem_frac"]


@beartype
def _add_latency_metrics(
    results: dict,
    data: dict,
    args,
    reference_ctms,
    step: int | None,
) -> None:
    """Populate results/data with latency metrics and dump JSON diagnostics."""
    # Measure emission and EOS latency with user perceived time
    latency_metrics, _, latencies = manage_ctm_export(
        args,
        results["user_perceived_timestamps"],
        reference_ctms,
        results["fnames"],
    )
    # Measure word timestamps with model timestamps
    _, timestamp_metrics, *_ = manage_ctm_export(
        args,
        results["model_timestamps"],
        reference_ctms,
        results["fnames"],
    )
    # Measure emission with finals timestamps
    final_latency_metrics, _, final_latencies = manage_ctm_export(
        args,
        results["finals_timestamps"],
        reference_ctms,
        results["fnames"],
    )

    dump_json = {
        "sil_frac": results.get("sil_frac"),
        "eos_frac": results.get("eos_frac"),
        "latency_metrics": latency_metrics,
        "final_latency_metrics": final_latency_metrics,
        "fw_latency": latencies.fw_latency,
        "sil_latency": latencies.sil_latency,
        "eos_latency": latencies.eos_latency,
        "latencies": latencies.raw_latency,
        "final_latencies": final_latencies.raw_latency,
    }
    if get_rank_or_zero() == 0:
        with open(f"{args.output_dir}/lats_{args.timestamp}_{step}.json", "w") as f:
            json.dump(dump_json, f, indent=2, ensure_ascii=False)

    data.update(latency_metrics)
    data.update(timestamp_metrics)
    results["latency_metrics"] = latency_metrics
    results["timestamp_metrics"] = timestamp_metrics
    results["final_latency_metrics"] = {
        f"final_{k}": v for k, v in final_latency_metrics.items()
    }


@beartype
@torch.no_grad()
def evaluate(
    epoch,
    step,
    val_loader: LhotsePipeline,
    detokenize,
    ema_model,
    joint_loss: JointLoss,
    decoder,
    cfg,
    args,
    error_rate: ErrorRate,
    calculate_loss=False,
    nth_batch_only=None,
    using_cpu=False,
    skip_logging=False,
    loss_mods: LossModifiers = IDENTITY_LOSS_MODIFIERS,
) -> Dict[str, Any]:
    """
    Perform evaluation.
    """
    ema_model.eval()
    joint_loss.eval()
    enc_time_reduction = ema_model.config.enc_stack_time_factor

    start_time = time.time()

    results = {
        "preds": [],
        "txts": [],
        "idx": [],
        "user_perceived_timestamps": [],
        "model_timestamps": [],
        "finals_timestamps": [],
        "wordstamps": [],
        "token_probs": [],
        "fnames": [],
    }
    if calculate_loss:
        results["losses"] = []

    total_loader_len = f"{len(val_loader):<10}"

    for i, (_, cuts, feats, tokens, _) in enumerate(val_loader):
        if nth_batch_only is not None:
            if i < nth_batch_only:
                continue
            elif i > nth_batch_only:
                break

        print(f"Evaluation: {i:>10}/{total_loader_len}", end="\r")

        assert tokens is not None

        # If the validation has requested CPU lhotse the data will not be on GPU yet
        if args.loader_val_device == "cpu" and not using_cpu:
            feats = feats.to("cuda", non_blocking=True)
            tokens = tokens.to("cuda", non_blocking=True)

        B = feats.data.size(1)
        device = feats.data.device
        fnames = [_infer_audio_path(cut) for cut in cuts.data]
        assert len(fnames) == B
        # The file names are absolute but the CTM files have relative paths,
        # hence we strip the data_dir prefix
        fnames = [_strip_data_dir(f, args.val_dataset_dir) for f in fnames]

        raw_transcripts = [_infer_transcript(cut) for cut in cuts.data]
        assert len(raw_transcripts) == B

        if calculate_loss:
            # --conditional is a train-only argument, so val.py's Namespace lacks it.
            if getattr(args, "conditional", False):
                prefix = broadcast_prefix(decoder.prefix_token_ids, B, device)
            else:
                prefix = None

            loss = val_loss(
                ema_model,
                joint_loss,
                feats=feats,
                tokens=tokens,
                loss_mods=loss_mods,
                prefix=prefix,
                joint_lattice_cap=args.joint_lattice_cap,
            )
            results["losses"] += gather_losses([loss.cpu()])

        pred, timestamps, probs, last_emit_time, o_width = run_rnnt_decoding(
            feats,
            decoder,
            cfg,
            args,
            enc_time_reduction,
            detokenize=detokenize,
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
            # Convert token timestamps to word timestamps

            kinds = {
                "user_perceived_timestamps": user_perceived_time,
                "model_timestamps": model_time,
                "finals_timestamps": final_time,
            }

            for k, f in kinds.items():
                results[k] += group_timestamps(
                    subwords,
                    [lmap(f, ts) for ts in timestamps],
                    preds,
                    last_emit_time,
                )

            initial_pad = initial_padding(args)

            results["wordstamps"] += [
                [
                    frame_to_time(
                        x,
                        o_width,
                        initial_pad=initial_pad,
                        head_offset=args.latency_head_offset,
                        tail_offset=args.latency_tail_offset,
                    )
                    for x in xs.seqs
                ]
                for xs in results["model_timestamps"]
            ]

    end_time = time.time()

    loss = process_evaluation_epoch(
        results,
        error_rate=error_rate,
        calculate_per=args.calculate_per,
        punctuation_marks=args.punctuation_marks,
        punctuation_normalization=args.punctuation_normalization,
        lang_code=args.val_standardizer_lang,
    )

    data = {"loss": loss, "took": end_time - start_time}
    _add_wer_metrics(results, data, results["all_wer_details"], error_rate)

    if args.calculate_per:
        _add_per_metrics(results, data, results["all_per_details"])

    if args.keywords_path and get_rank_or_zero() == 0:
        _add_keyword_metrics(results, data, args)

    if results.get("user_perceived_timestamps") and get_rank_or_zero() == 0:
        _add_timestamp_fractions(results, data)

    if args.calculate_emission_latency:
        # In distributed setup, all ranks need to execute this (check manage_ctm_export)
        ref_manifests = get_reference_manifest_files(args)
        _add_latency_metrics(results, data, args, ref_manifests, step)

    if not skip_logging:
        log((epoch,), step, "dev_ema", data)

    if get_rank_or_zero() == 0:
        _dump_preds_to_json(
            results["all_wer_details"],
            results["all_per_details"],
            results["preds"],
            results["txts"],
            results["fnames"],
            error_rate,
            args.calculate_per,
            output_path=f"{args.output_dir}/preds_{args.timestamp}_{step}.json",
        )

    ema_model.train()
    joint_loss.train()
    return results


@torch.no_grad()
@beartype
def run_rnnt_decoding(
    feats: TimeFirst,
    decoder,
    cfg: dict,
    args: Namespace,
    enc_time_reduction: int | float,
    detokenize,
):
    """
    Run core RNN-T decoding pipeline.
    """
    # If state resets is on, prepare the feats
    if args.sr_segment:
        feats, meta = state_resets_reshape_batched_feats(
            args.sr_segment, args.sr_overlap, cfg, feats
        )

    responses = decoder.decode(feats)

    # All of these are as-reported by the finals
    pred, model_t, final_t, probs = split_batched_finals(responses)

    # Makes timestamps optimistic on partials
    responses = lmap(fuse_partials, responses)
    # We can guarantee (emit, model) <= final
    _, emit_t, _, _ = split_batched_finals(responses)

    timestamps = lstarmap_zip(
        lambda *x: lstarmap_zip(FullStamp, *x), model_t, emit_t, final_t
    )

    # If state resets is used, merge predictions and timestamps of segments into one
    if args.sr_segment:
        pred, timestamps, probs = state_resets_merge_batched_segments(
            pred,
            timestamps,
            probs,
            enc_time_reduction,
            meta,
            decoder.eos_index if args.eos_is_terminal else None,
            detokenize=detokenize,
        )

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
        initial_pad=initial_padding(args),
        feat_lens=feats.lens,
        eos_vad_threshold=args.eos_vad_threshold,
        eos_info=eos_info,
    )
    return pred, timestamps, probs, last_emit_time, o_width
