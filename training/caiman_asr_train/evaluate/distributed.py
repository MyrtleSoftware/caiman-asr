import torch
import torch.distributed as dist
from beartype import beartype
from beartype.typing import Dict, Optional, Tuple

from caiman_asr_train.evaluate.metrics import word_error_rate


@beartype
def process_evaluation_epoch(
    aggregates: Dict[str, list], standardize_wer: bool
) -> Tuple[float, Optional[float]]:
    """
    Processes results from each worker at the end of evaluation and combine to final result

    Aggregates will be updated in-place to have timestamps from all processes.

    Args:
        aggregates: dictionary containing information of entire evaluation
        standardize_wer: whether to apply Whisper normalizatio rules to
        the transcripts
    Return:
        wer: final word error rate
        loss: final loss
    """
    if "losses" in aggregates:
        eloss = torch.mean(torch.stack(aggregates["losses"])).item()
    else:
        eloss = None

    hypotheses = aggregates["preds"]
    references = aggregates["txts"]

    wer, scores, num_words = word_error_rate(
        hypotheses, references, standardize=standardize_wer
    )
    multi_gpu = dist.is_initialized()
    if multi_gpu:
        if eloss is not None:
            eloss /= dist.get_world_size()
            eloss_tensor = torch.tensor(eloss).cuda()
            dist.all_reduce(eloss_tensor)
            eloss = eloss_tensor.item()

        scores_tensor = torch.tensor(scores).cuda()
        dist.all_reduce(scores_tensor)
        scores = scores_tensor.item()
        num_words_tensor = torch.tensor(num_words).cuda()
        dist.all_reduce(num_words_tensor)
        num_words = num_words_tensor.item()
        wer = scores * 1.0 / num_words

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
