import torch
from beartype import beartype
from beartype.typing import List, Optional, Tuple

from caiman_asr_train.evaluate.state_resets.core import (
    get_state_resets_stats,
    state_resets_merge_segments,
    state_resets_reshape_feats,
)
from caiman_asr_train.evaluate.state_resets.timestamp import Timestamp


@beartype
def state_resets_reshape_batched_feats(
    sr_segment: float,
    sr_overlap: float,
    cfg: dict,
    feats: torch.Tensor,
    feat_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int, int]]]:
    """Reshape the feats tensor with length of first dimension equal to the segment.

    This function will reshape the feats and feat_lens tensors to reflect the use of
    State Resets. Long utterances are divided into segments of sr_segment_frames and
    redistributed across the batch dimension.

    Parameters
    ----------
    sr_segment
        state resets segment duration in seconds
    sr_overlap
        state resets overlap duration in seconds
    cfg
        dictionary with configuration arguments
    feats
        tensor of feats with size [seq_len, batch_size, input]
    feat_lens
        tensor of length of feats with size [batch_size]

    Returns
    -------
    feats
        tensor of new feats with size [<= sr_segment_frames, >=batch_size, input]
    feat_lens
        tensor of length feats with size [>=batch_size]
    meta
        meta data for reversing this operation
    """

    jagged_feats = []
    jagged_lens = []
    meta = []

    # feats is:     (seq_len, batch, hidden)
    # feat_lens is: (batch, )

    sr_segment_frames, sr_overlap_frames = get_state_resets_stats(
        sr_segment, sr_overlap, cfg
    )

    for b in range(feats.shape[1]):
        feat_len = feat_lens[b].unsqueeze(0)
        feat = feats[:feat_len, b, :].unsqueeze(1)

        (feat, feat_len, _, _) = state_resets_reshape_feats(
            sr_segment, sr_overlap, cfg, feat, feat_len
        )

        for bb in range(feat.shape[1]):
            jagged_feats.append(feat[:, bb, :])
            jagged_lens.append(feat_len[bb])

        meta.append((feat.shape[1], sr_segment_frames, sr_overlap_frames))

    # This is stack + pad in one call
    feats = torch.nn.utils.rnn.pad_sequence(jagged_feats)
    # All the same length so can use regular stack
    feat_lens = torch.stack(jagged_lens)

    return feats, feat_lens, meta


@beartype
def state_resets_merge_batched_segments(
    pred: List[List[int]],
    timestamps: List[List[Timestamp]],
    probs: List[List[float]],
    enc_time_reduction: int,
    meta: List[Tuple[int, int, int]],
    eos_idx: Optional[int],
) -> Tuple[List[List[int]], List[List[Timestamp]], Optional[List[List[float]]]]:
    """Return prediction tokens and timestamps for all segments concatenated.

    This function will return the prediction tokens and the timestamps without
    the overlap and in a way that appears as the segment tokens were not decoded
    independently.

    Parameters
    ----------
    pred
        list with lists of tokens per segment
    timestamps
        list with lists of decoder timestamps per segment
    probs
        list of lists of per-token probabilities for each segment
    enc_time_reduction
        factor for time reduction in the encoder
    meta
        meta data from state_resets_reshape_batched_feats
    eos_idx
        index of the end of sequence token, if this is not None then merging
        will stop at the first occurrence of this token.

    Returns
    -------
    pred
        list with list of tokens for concatenated segments
    timestamps
        list with list of decoder timestamps for concatenated segments
    probs
        list of list of per-token probabilities for concatenated segments or None
    """

    o_pred = []
    o_timestamps = []
    o_probs = []

    acc = 0

    for i, sr_segment_frames, sr_overlap_frames in meta:
        l_pred = pred[acc : acc + i]
        l_timestamps = timestamps[acc : acc + i]
        l_probs = probs[acc : acc + i]

        if eos_idx is not None:
            for j, pr in enumerate(l_pred):
                if eos_idx in pr:
                    break

            l_pred = l_pred[: j + 1]
            l_timestamps = l_timestamps[: j + 1]
            l_probs = l_probs[: j + 1]

        l_pred, l_timestamps, l_probs = state_resets_merge_segments(
            l_pred,
            l_timestamps,
            l_probs,
            enc_time_reduction,
            sr_segment_frames,
            sr_overlap_frames,
        )

        o_pred.extend(l_pred)
        o_timestamps.extend(l_timestamps)

        if l_probs:
            o_probs.extend(l_probs)

        acc += i

    assert len(o_pred) == len(meta)

    return o_pred, o_timestamps, o_probs if l_probs else None
