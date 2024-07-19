import torch
import torch.distributed as dist
from beartype import beartype
from beartype.typing import Dict, Optional, Tuple

from caiman_asr_train.evaluate.distributed_utils import multigpu_wer, sum_across_gpus
from caiman_asr_train.evaluate.wer_breakdown import print_wer_breakdown


@beartype
def process_evaluation_epoch(
    aggregates: Dict[str, list],
    standardize_wer: bool,
    breakdown_wer: bool,
    breakdown_chars: str,
) -> Tuple[float, Optional[float]]:
    """
    Processes results from each worker at the end of evaluation and combine to final result

    Aggregates will be updated in-place to have timestamps from all processes.

    Args:
        aggregates: dictionary containing information of entire evaluation
        standardize_wer: whether to apply Whisper normalization rules to
        the transcripts
    Return:
        wer: final word error rate
        loss: final loss
    """
    if "losses" in aggregates:
        eloss = torch.mean(torch.stack(aggregates["losses"])).item()
    else:
        eloss = -1.0

    hypotheses = aggregates["preds"]
    references = aggregates["txts"]

    wer = multigpu_wer(hypotheses, references, standardize_wer)
    if breakdown_wer:
        print_wer_breakdown(hypotheses, references, breakdown_chars)

    multi_gpu = dist.is_initialized()
    if multi_gpu:
        eloss /= dist.get_world_size()
        eloss = sum_across_gpus(eloss)

        gathered_results = (
            [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        )
        # Gather aggregates dictionary across all workers
        dist.gather_object(aggregates, gathered_results, dst=0)

        # If rank 0 process, combine timestamp data from all processes
        if dist.get_rank() == 0:
            lst_seq_time = [
                timestamp
                for results in gathered_results
                for timestamp in results["timestamps"]
            ]
            aggregates["timestamps"] = lst_seq_time
    return wer, eloss
