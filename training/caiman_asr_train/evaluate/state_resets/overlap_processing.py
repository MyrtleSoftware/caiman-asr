import string
import warnings
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from beartype import beartype
from beartype.typing import Callable, List, Optional, Tuple, TypeVar, Union
from Levenshtein import distance

from caiman_asr_train.evaluate.state_resets.timestamp import (
    Timestamp,
    add_frames,
)


@beartype
def process_time(
    timestamps: List[List[Timestamp]],
    enc_time_reduction: int,
    segment_frames: int,
    overlap_frames: int,
) -> List[Timestamp]:
    """Adjusts segmented decoder timestamps to simulate continuous decoding.

    This function converts segmented timestamps to a continuous timeline by
    incrementing each subsequent segment's timestamps. The increment is based on
    the net duration of the previous segments, accounting for overlaps. The
    result is a single list of timestamps as if the utterance was decoded in one
    go.

    For example:
    >>> process_time([[1, 3, 5, 6, 10], [2, 3, 5, 7, 8], [3, 4]],
    ... enc_time_reduction=2, segment_frames=26, overlap_frames=6)
    [1, 3, 5, 6, 10, 12, 13, 15, 17, 18, 23, 24]

    Parameters
    ----------
    timestamps
        Nested list of decoder timestamps for each segment.
    enc_time_reduction
        Factor by which encoder reduces time resolution.
    segment_frames
        Number of frames in each segment.
    overlap_frames
        Number of frames in overlap.

    Returns
    -------
    time_shifted
        List of ints representing the adjusted, continuous timestamps.
    """
    if (segment_frames - overlap_frames) % enc_time_reduction != 0:
        warnings.warn(
            f"{segment_frames=} - {overlap_frames=} "
            f"must be divisible by {enc_time_reduction=} "
            "in order to have accurate integer timestamps"
        )
    max_time_per_segment = (segment_frames - overlap_frames) // enc_time_reduction
    time_shifted = timestamps[0]

    for itm, lst in enumerate(timestamps[1:]):
        items = [add_frames(t, (itm + 1) * max_time_per_segment) for t in lst]
        time_shifted.extend(items)

    return time_shifted


@beartype
def best_split(lhs: list[tuple[int, str]], rhs: list[tuple[int, str]], k: int = 3):
    """
    Given the primary stream of tokens 'lhs', and the secondary stream 'rhs'.
    Find the partition of `lhs : z | n | m` and rhs : `x | y | _` such that
    the concatenation `merged : z | n | _` is continuous speech.

    Arguments:
        lhs: List of tuples (index, token) for the primary stream.
        rhs: List of tuples (index, token) for the secondary stream.

    Returns:
        tuple (best, n, m, x, y) where:
            best: The best score for the partition.
               n: the number of tokens in the primary stream before the partition.
               m: the number of tokens in the primary stream after the partition.
               x: the number of tokens in the secondary stream before the partition.
               y: the number of tokens in the secondary stream after the partition.

        Also see descriptions of n, m, x, y in the docstring above.
    """

    # This parameter is kept fixed to align with the streaming implementation
    # which can't backtrack on the primary stream.
    m = 0

    best, N, M, X, Y = 0, 0, 0, 0, 0

    lhs_toks = [x for (_, x) in lhs]
    rhs_toks = [x for (_, x) in rhs]

    for x in range(min(10, len(lhs_toks))):
        for n in range(len(lhs_toks) - m):
            z = len(lhs_toks) - n - m
            lhs_s = "".join(lhs_toks[z : z + n])
            # Search the fat-diagonal
            for y in range(max(0, n - k), min(len(rhs_toks) - x, n + k - 1)):
                rhs_s = "".join(rhs_toks[x : x + y])
                d = distance(lhs_s, rhs_s)
                s = n - m - x + y - d
                if s > best:
                    best = s
                    N = n
                    M = m
                    X = x
                    Y = y

    return best, N, M, X, Y


T = TypeVar("T")


@beartype
def denorm(tok: T, detokenize: Callable[[T], str]) -> str:
    """
    De-tokenize with primitive normalization.
    """
    seg = detokenize(tok)

    # Catch special tokens
    if seg.startswith("<") and seg.endswith(">"):
        return ""

    seg = seg.translate(str.maketrans("", "", string.punctuation))
    seg = seg.strip().lower()

    return seg


@beartype
def get_unique_predictions(
    pred: List[List[int]],
    timestamps: List[List[Timestamp]],
    probs: List[List[float]],
    detokenize: Callable[[int], str],
) -> Tuple[List[List[int]], List[List[Timestamp]], Optional[List[List[float]]]]:
    """
    Return transcripts without the overlapping segment.

    This function receives the segments of the transcripts and their
    corresponding timestamps/probs. It then scans the transcripts and removes
    the overlapping region with a levenshtein based heuristic. If per-token
    probabilities are provided, then the corresponding probabilities are also
    removed.

    For example, assume the following example input:

    pred:        [[7, 2, 3, 6, 5], [2, 6, 5, 9, 7]]
    timestamps:  ...
    probs:       ...

    The output of the function will be:

    pred:        [[7, 2, 3, 6, 5], [9, 7]]
    timestamps:  ...
    probs:       ...

    where the second segment has the first 3 tokens ("2", "6" and "5") removed.

    Parameters
    ----------
    pred
        Segmented transcript tokens, grouped per segment.
    timestamps
        Corresponding timestamps for the tokens, grouped per segment.
    probs
        Per-token probabilities, grouped per segment.
    enc_time_reduction
        Time stacking factor.
    overlap_frames
        Number of frames in the overlap region.
    lookahead
        The number of tokens to be examined for commonalities with the last
        few tokens of the previous segment.

    Returns:
    --------
    Tuple containing:
    - List of transcript segments with overlap removed.
    - List of adjusted timestamps corresponding to the transcripts.
    - List of adjusted probabilities corresponding to the transcripts or None
    """

    assert detokenize is not None

    def as_words(toks):
        return [
            (i, s) for i, tok in enumerate(toks) if (s := denorm(tok, detokenize)) != ""
        ]

    # Spawn a separate CPU process for best_split (levenshtein), timeout for 60s
    with ProcessPoolExecutor(max_workers=1) as executor:
        for i in range(len(pred) - 1):
            acc = as_words(pred[i])
            nxt = as_words(pred[i + 1])

            if len(acc) == 0 or len(nxt) == 0:
                continue

            future = executor.submit(best_split, acc, nxt)

            try:
                _, _, m, x, y = future.result(timeout=60.0)
            except TimeoutError:
                executor.shutdown(wait=False, cancel_futures=True)
                raise TimeoutError(f"best_split exceeded timeout at segment {i}")

            # Discard last m from acc,
            # Discard first x + y from nxt

            keep_lhs_idx, _ = acc[len(acc) - m - 1]
            keep_rhs_idx, _ = nxt[0 if x + y == 0 else x + y - 1]

            pred[i] = pred[i][: keep_lhs_idx + 1]
            pred[i + 1] = pred[i + 1][keep_rhs_idx + 1 :]

            timestamps[i] = timestamps[i][: keep_lhs_idx + 1]
            timestamps[i + 1] = timestamps[i + 1][keep_rhs_idx + 1 :]

            if probs:
                probs[i] = probs[i][: keep_lhs_idx + 1]
                probs[i + 1] = probs[i + 1][keep_rhs_idx + 1 :]

    return pred, timestamps, probs


@beartype
def combine_predictions(
    pred_list: List[List[Union[float, int]]],
) -> List[List[Union[float, int]]]:
    """Flatten a list of lists.

    When state resets during inference is used, the
    predictions and the timestamps are in a list of
    lists. This function returns a list of size 1
    containing a flattened list.

    Parameters
    ----------
    pred_list
        list of lists with predicted tokens or timestamps

    Returns
    -------
        list of flattened list
    """
    return [[item for sublist in pred_list for item in sublist]]
