#!/usr/bin/env python3

import torch
import torch.distributed as dist
from beartype import beartype

from caiman_asr_train.evaluate.error_rates import ErrorRate
from caiman_asr_train.evaluate.metrics import word_error_rate


def sum_across_gpus(number):
    number_tensor = torch.tensor(number).cuda()
    dist.all_reduce(number_tensor)
    return number_tensor.item()


@beartype
def sync_wer_across_gpus(wer_tuple: tuple[float, int, int]) -> float:
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
    hypotheses: list[str],
    references: list[str],
    error_rate: ErrorRate,
    standardize: bool,
) -> float:
    wer_tuple = word_error_rate(hypotheses, references, error_rate, standardize)
    return sync_wer_across_gpus(wer_tuple)
