import torch
import torch.distributed as dist
from beartype import beartype
from beartype.typing import Dict, Optional

from caiman_asr_train.evaluate.distributed_utils import sum_across_gpus
from caiman_asr_train.evaluate.error_rates import ErrorRate
from caiman_asr_train.evaluate.metrics import (
    PerDetails,
    punctuation_error_rate,
    word_error_rate,
)


@beartype
def sync_aggregates_across_ranks(aggregates: Dict[str, list]) -> None:
    """
    All-gather aggregates so every rank holds the concatenated results from all peers.
    """
    if not dist.is_initialized():
        return
    gathered_results = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_results, aggregates)

    for x in [
        "user_perceived_timestamps",
        "model_timestamps",
        "finals_timestamps",
        "wordstamps",
        "preds",
        "txts",
        "fnames",
        "all_wer_details",
        "all_per_details",
    ]:
        if x in aggregates:
            combined_data = [
                data for results in gathered_results for data in results[x]
            ]
            aggregates[x] = combined_data


@beartype
def process_evaluation_epoch(
    aggregates: Dict[str, list],
    lang_code: str,
    error_rate: ErrorRate,
    calculate_per: bool,
    punctuation_marks: list[str],
    punctuation_normalization: bool = True,
) -> Optional[float]:
    """
    Processes results from each worker at the end of evaluation and combine to final result

    Aggregates will be updated in-place to have timestamps, model_timestamps,
    finals_timestamps, wordstamps, preds, txts, fnames, all_wer_details,
    all_per_details from all processes.

    Args:
        aggregates: dictionary containing information of entire evaluation
        the transcripts
        lang_code: language code of the transcripts
        error_rate: type of error rate to compute (WORD/CHAR/MIXTURE)
        punctuation_marks: list of punctuation marks to consider for PER
        punctuation_normalization: whether to normalize punctuation for PER
    Return:
        loss: final loss
    """
    if "losses" in aggregates:
        eloss = torch.mean(torch.stack(aggregates["losses"])).item()
    else:
        eloss = -1.0

    hypotheses = aggregates["preds"]
    references = aggregates["txts"]

    all_wer_details = []
    all_per_details = []

    # Compute local WER/PER for this rank's data
    for hyp, ref in zip(hypotheses, references):
        wer_detail = word_error_rate([hyp], [ref], error_rate, lang_code)
        all_wer_details.append(wer_detail)

        if calculate_per:
            per_detail = punctuation_error_rate(
                [hyp], [ref], punctuation_marks, punctuation_normalization
            )
        else:
            # Placeholder if PER is not calculated
            per_detail = PerDetails(-1.0, 0, 0)
        all_per_details.append(per_detail)
    aggregates["all_wer_details"] = all_wer_details
    aggregates["all_per_details"] = all_per_details

    multi_gpu = dist.is_initialized()
    if multi_gpu:
        eloss = sum_across_gpus(eloss / dist.get_world_size())
        sync_aggregates_across_ranks(aggregates)

    return eloss
