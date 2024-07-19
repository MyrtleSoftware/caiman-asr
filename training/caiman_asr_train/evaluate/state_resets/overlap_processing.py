import warnings
from math import ceil

from beartype import beartype
from beartype.typing import List, Optional, Tuple, Union


@beartype
def process_time(
    timestamps: List[List[int]],
    enc_time_reduction: int,
    segment_frames: int,
    overlap_frames: int,
) -> List[int]:
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
        items = [i + (itm + 1) * max_time_per_segment for i in lst]
        time_shifted.extend(items)

    return time_shifted


@beartype
def get_unique_predictions(
    pred: List[List[int]],
    timestamps: List[List[int]],
    probs: List[List[float]],
    enc_time_reduction: int,
    overlap_frames: int,
    lookahead: int = 3,
) -> Tuple[List[List[int]], List[List[int]], Optional[List[List[float]]]]:
    """Return transcripts without the overlapping segment.

    This function receives the segments of the transcripts and their corresponding
    timestamps. It then scans the timestamps, and the ones
    that are less than or equal to the number of overlap frames are dropped, along
    with the respective tokens. This is happening because the tokens belong to the
    overlapping region. Furthermore, the first "lookahead" number of
    tokens is scanned, and if any of these tokens are among the last few tokens
    of the previous segment, they are dropped.

    If per-token probabilities are provided, then the corresponding probabilities are
    also removed.

    For example, assume the following example input:
    pred:        [[7, 2, 3, 6, 5], [2, 6, 5, 9, 7]]
    timestamps:  [[1, 2, 3, 4, 6], [1, 3, 4, 5, 6]]
    overlap_frames: 2
    lookahead: 3
    The output of the function will be:

    will return:
    pred:        [[7, 2, 3, 6, 5], [9, 7]]
    timestamps:  [[1, 2, 3, 4, 6], [5, 6]]
    where the second segment has the first 3 tokens ("2", "6" and "5") removed.
    The first token ("2") is removed because it belongs to the overlap region
    (timestamp=1 where the overlap duration is 2 frames). Then, the last lookahead tokens
    of the previous segment ( "3", "6" and "5") are kept in a list of trusted tokens.
    The length of this list is up to the lookahead number. Then, the first lookahead
    number of tokens in the current segment (which is [6, 5, 9, 7] after dropping the token
    that belongs to the overlapping region) are scanned for tokens that exist in the
    trusted list. The common tokens "6" and "5" are removed.

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

    # alter the duration to match the stack time
    overlap_dur = ceil(overlap_frames / enc_time_reduction)

    unique_pred = [pred[0]]
    unique_timestamps = [timestamps[0]]
    unique_probs = [probs[0]] if probs else None

    for seg, (curr_pred, curr_timestamps) in enumerate(
        zip(pred[1:], timestamps[1:], strict=True), start=1
    ):
        # Decide how many tokens to omit from the start of the current segment
        to_omit = 0
        for timestamp in curr_timestamps:
            if timestamp < overlap_dur:
                to_omit += 1
            else:
                break

        # Append the modified segment predictions and timestamps, omitting the
        # calculated number of elements
        unique_pred.append(curr_pred[to_omit:])
        unique_timestamps.append(curr_timestamps[to_omit:])
        if probs:
            unique_probs.append(probs[seg][to_omit:])

    # after removing overlap, scan for common tokens at the boundary
    remove_common, adjusted_timestamps, adjusted_probs = (
        [unique_pred[0]],
        [unique_timestamps[0]],
        [unique_probs[0]] if unique_probs else None,
    )

    for seg in range(1, len(unique_pred)):
        trusted_list = unique_pred[seg - 1][-lookahead:]

        # Remove the trusted tokens from the current segment
        (
            unique_pred_segm,
            unique_pred_time,
            unique_prob_segm,
        ) = manage_boundary_common_tokens(
            segm=unique_pred[seg],
            t_st=unique_timestamps[seg],
            probs=unique_probs[seg] if unique_probs else None,
            trusted_list=trusted_list,
            lookahead=lookahead,
        )
        remove_common.append(unique_pred_segm)
        adjusted_timestamps.append(unique_pred_time)
        if probs:
            adjusted_probs.append(unique_prob_segm)

    return remove_common, adjusted_timestamps, adjusted_probs


@beartype
def manage_boundary_common_tokens(
    segm: List[int],
    t_st: List[int],
    probs: Optional[List[float]],
    trusted_list: List[int],
    lookahead: int,
) -> Tuple[List[int], List[int], Optional[List[float]]]:
    """
    Return segment tokens, timestamps, and probs without duplicates from previous segment.

    This function cleans the duplicate tokens in the current segment, based on the
    last lookahead number of tokens found in the previous segment.
    This method is streaming compatible, as every token can be checked independently
    of the previous and following tokens.

    Parameters
    ----------
    segm
        list of tokens in the current segment
    t_st
        timestamps for tokens in the current segment
    probs
        probabilities for tokens in the current segment
    trusted_list
        list of up to lookahead number of tokens from previous segment
    lookahead
        number of tokens to check in the current segment if exist in trusted list
    """
    for token in segm[:lookahead]:
        if token in trusted_list:
            # remove token & timestamp from the segment list & timestamp list respectively
            t_st.pop(segm.index(token))
            if probs:
                probs.pop(segm.index(token))
            segm.remove(token)
            # get the token index in the trusted list and update it
            index_trusted = trusted_list.index(token)
            trusted_list = trusted_list[index_trusted + 1 :]
    return segm, t_st, probs


@beartype
def combine_predictions(
    pred_list: List[List[Union[float, int]]]
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
