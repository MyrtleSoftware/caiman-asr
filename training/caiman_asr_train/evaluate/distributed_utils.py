#!/usr/bin/env python3

import torch
import torch.distributed as dist
from beartype import beartype
from beartype.typing import List, Tuple

from caiman_asr_train.evaluate.error_rates import ErrorRate
from caiman_asr_train.evaluate.metrics import punctuation_error_rate, word_error_rate


def sum_across_gpus(number):
    number_tensor = torch.tensor(number).cuda()
    dist.all_reduce(number_tensor)
    return number_tensor.item()


@beartype
def sync_wer_across_gpus(wer_tuple: Tuple[float, int, int]) -> float:
    if dist.is_initialized():
        _, scores, num_words = wer_tuple
        scores = sum_across_gpus(scores)
        num_words = sum_across_gpus(num_words)
        wer = scores * 1.0 / num_words
    else:
        wer, _, _ = wer_tuple
    return wer


@beartype
def multigpu_wer(
    hypotheses: List[str],
    references: List[str],
    error_rate: ErrorRate,
    standardize: bool,
) -> float:
    wer_tuple = word_error_rate(hypotheses, references, error_rate, standardize)
    return sync_wer_across_gpus(wer_tuple)


@beartype
def multigpu_per(
    hypotheses: List[str],
    references: List[str],
    punctuation_marks: List[str] = [",", ".", "?"],
    punctuation_normalization: bool = True,
) -> float:
    """
    Compute PER across multiple GPUs by aggregating counts.

    We compute local (per, pun_symbols), convert to local integer error counts as
    per_count = round(per * pun_symbols) to match implementation, then sum both
    per_count and pun_symbols across GPUs, and finally divide.
    """
    per_local, pun_symbols_local = punctuation_error_rate(
        hypotheses, references, punctuation_marks, punctuation_normalization
    )

    if dist.is_initialized():
        # Round to convert local PER to an integer error count
        # to mirror single-GPU semantics
        # see `punctuation_error_rate` in `metrics.py`
        per_count_local = round(per_local * pun_symbols_local)
        per_count_global = sum_across_gpus(per_count_local)
        pun_symbols_global = sum_across_gpus(pun_symbols_local)
        if pun_symbols_global > 0:
            return per_count_global / pun_symbols_global
        else:
            return 0.0
    else:
        return per_local
