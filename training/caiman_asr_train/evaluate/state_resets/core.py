import torch
from beartype import beartype
from beartype.typing import List, Optional, Tuple
from jaxtyping import Float, jaxtyped

from caiman_asr_train.evaluate.state_resets.overlap_processing import (
    combine_predictions,
    get_unique_predictions,
    process_time,
)
from caiman_asr_train.evaluate.state_resets.timestamp import Timestamp
from caiman_asr_train.utils.frame_width import input_feat_frame_width


@beartype
def state_resets_reshape_feats(
    sr_segment: float,
    sr_overlap: float,
    cfg: dict,
    feats: torch.Tensor,
    feat_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Reshape the feats tensor with length of first dimension equal to the segment.

    This function will reshape the feats tensor to reflect the use of State Resets.
    The first two dimensions of the feats tensor have sizes equal to sequence length
    and batch size equal to 1, respectively. After reshaping the feats the dimensions
    will be number of frames per segment and number of segments respectively.

    Parameters
    ----------
    sr_segment
        state resets segment duration in seconds
    sr_overlap
        state resets overlap duration in seconds
    cfg
        dictionary with configuration arguments
    feats
        tensor of feats with size [seq_len, batch_size=1, input]
    feat_lens
        tensor of length of feats with size [batch_size]

    Returns
    -------
    feats
        tensor of feats with size [sr_segment_frames, n_segments, input]
    feat_lens
        tensor of length feats with size [n_segments]
    sr_segment_frames
        number of frames of the segment
    sr_overlap_frames
        number of frames of the overlap
    """
    sr_segment_frames, sr_overlap_frames = get_state_resets_stats(
        sr_segment, sr_overlap, cfg
    )

    feats, feat_lens = get_state_resets_feats(
        feats, feat_lens, sr_segment_frames, sr_overlap_frames
    )

    return feats, feat_lens, sr_segment_frames, sr_overlap_frames


@beartype
def state_resets_merge_segments(
    pred: List[List[int]],
    timestamps: List[List[Timestamp]],
    probs: List[List[float]],
    enc_time_reduction: int,
    sr_segment_frames: int,
    sr_overlap_frames: int,
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
    sr_segment_frames
        number of frames of the segment
    sr_overlap_frames
        number of frames of the overlap

    Returns
    -------
    pred
        list with list of tokens for concatenated segments
    timestamps
        list with list of decoder timestamps for concatenated segments
    probs
        list of list of per-token probabilities for concatenated segments or None
    """
    pred, timestamps, probs = get_unique_predictions(
        pred,
        timestamps,
        probs,
        enc_time_reduction,
        sr_overlap_frames,
        lookahead=3,
    )
    # combine segments into single list
    pred = combine_predictions(pred)
    if probs:
        probs = combine_predictions(probs)

    # transform timestamps to reflect continues decoding
    timestamps = process_time(
        timestamps, enc_time_reduction, sr_segment_frames, sr_overlap_frames
    )

    return pred, [timestamps], probs


@beartype
def validate_state_resets_arguments(sr_segment: float, sr_overlap: float) -> None:
    """Check the values provided by the user.

    Parameters
    ----------
    sr_segment
        segment duration in seconds
    sr_overlap
        overlap duration between segments in seconds

    Raises
    ------
    ValueError
    """
    if sr_segment <= 0 or sr_overlap < 0:
        raise ValueError(
            "Please ensure you provide positive --sr_segment and non-negative "
            "--sr_overlap to use State Resets."
        )

    if sr_segment <= sr_overlap:
        raise ValueError(
            "Please ensure that --sr_segment is greater than --sr_overlap when using "
            "State Resets."
        )


@beartype
def get_state_resets_stats(
    sr_segment: float, sr_overlap: float, model_config: dict
) -> Tuple[int, int]:
    """Converts durations for segments and overlaps into number of frames.

    This function receives segment duration for the state resets and the overlap
    duration (which can be zero). It converts the aforementioned durations from
    seconds into number of frames. The frame duration is extracted from the parsed
    config file.

    Parameters
    ----------
    sr_segment
        segment duration in seconds
    sr_overlap
        overlap duration between segments in seconds
    model_config
        the model configuration dictionary

    Returns
    -------
    sr_segment_frames
        number of frames of features in the segment
    sr_overlap_frames
        number of frames of features in the overlapping region
    """

    frame_duration = input_feat_frame_width(model_config)

    validate_state_resets_arguments(sr_segment, sr_overlap)

    # convert seconds into time frames
    sr_segment_frames = round(sr_segment / frame_duration)
    sr_overlap_frames = round(sr_overlap / frame_duration)

    return sr_segment_frames, sr_overlap_frames


@beartype
def get_state_resets_feats(
    feats: torch.Tensor,
    feat_lens: torch.Tensor,
    sr_segment_frames: int,
    sr_overlap_frames: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the features split into segments containing overlapping regions.

    Returns a new tensor with the features containing appropriate padding at the end.

    For instance suppressing the third dimension, the original feats tensor is "
    "[seq_len, batch=1] which will be transformed into [sr_segment_frames, n_segments].
    The sr_segment_frames has a length such that
    (sr_segment_frames * n_segments) % (seq_len + padding) == 0.

    Parameters
    ----------
    feats
        the audio features tensor with size [seq_len, batch_size, input]
    feat_lens
        the audio features size per batch, size is [batch_size]
    sr_segment_frames
        number of frames of features in segment
    sr_overlap_frames
        number of frames of features in the overlapping region

    Returns
    -------
    new_feats
        the segmented audio features with size [sr_segment_frames, n_segments, input]
    new_feat_lens
        the audio features with size [n_segments]
    """

    # assert that validation batch size=1.
    assert (
        feats.shape[1] == 1
    ), f"feats with size {feats.size()} are batched, set --val_batch_size=1"

    # if size of feats (= seq_len) < segment frames do not apply state resets
    if feats.shape[0] < sr_segment_frames:
        return feats, feat_lens

    # get the number of segments and the padding that needs to be added
    n_segments, padding = get_segmenting_info(
        all_frames=feat_lens.data.item(),
        overlap_frames=sr_overlap_frames,
        segment_frames=sr_segment_frames,
    )

    # add padding
    padded_feats = torch.nn.functional.pad(feats, (0, 0, 0, 0, 0, padding))

    # split into segments in the second dimension
    if sr_overlap_frames > 0:
        new_feats = extend_feats_with_overlaps(
            padded_feats=padded_feats,
            overlap_frames=sr_overlap_frames,
            segment_frames=sr_segment_frames,
            n_segments=n_segments,
        )
    else:  # sr_overlap_frames == 0
        new_feats = padded_feats

    # get correct batching for decoder that will do the state resets
    new_feats, new_feat_lens = reshape_feats(new_feats, n_segments)

    return new_feats, new_feat_lens


@beartype
def reshape_feats(
    feats: torch.Tensor, n_segments: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reshape feats tensor - stack in the batch dimension

    This function will split the tensor and stack it in the second dimension (batch) so
    that the decoder will process each batch as independent entity.

    For example, the new_feats tensor:
    [[ A B C C D E E F G G H 0]]
    will be reshaped into:
    [[A B C] [C D E] [E F G] [G H 0]]

    Parameters
    ----------
    feats
        the audio features tensor with size [seq_len, batch_size=1, input]
    n_segments
        number of segments the feats will be split into

    Returns
    -------
    new_feats
        the segmented audio features with size [sr_segment_frames, n_segments, input]
    new_feat_lens
        the audio features [n_segments]
    """
    # get correct batching for decoder that will do the state resets

    feats = feats.transpose(0, 1)
    feats = feats.reshape(n_segments, -1, feats.shape[2])
    new_feats = feats.transpose(0, 1)

    new_feat_lens = (
        torch.ones(new_feats.shape[1], dtype=torch.int32, device=new_feats.device)
        * new_feats.shape[0]
    )

    return new_feats, new_feat_lens


@jaxtyped(typechecker=beartype)
def extend_feats_with_overlaps(
    padded_feats: Float[torch.Tensor, "seq_len batch_size input"],
    overlap_frames: int,
    segment_frames: int,
    n_segments: int,
) -> Float[torch.Tensor, "{segment_frames*n_segments} batch_size input"]:
    """Extend the feats tensor with overlapping regions

    This function will return a tensor that contains the overlapping
    regions in a single dimension

    For example, for the following parameters:
        * overlap_frames=1
        * segment_frames=3
        * unique_frames= segment_frames - overlap_frames = 2
        * n_segments=4

    The padded_feats will be:
    [[ A B C D E F G H 0]]

    and will be transformed into:
    [[ A B C C D E E F G G H 0 ]]

    Parameters
    ----------
    padded_feats
        the audio features tensor with size [seq_len, batch_size, input]
    overlap_frames
        number of frames of features in the overlapping region
    segment_frames
        number of frames of features in segment
    n_segments
        how many segments of specific duration will exist

    Returns
    -------
    new_feats
        the segmented audio features with size
        [segment_frames*n_segments, batch_size, input]
    """
    # first generate the overlapping region starting with the first
    unique_frames = segment_frames - overlap_frames

    pieces = [
        padded_feats[unique_frames * i : segment_frames + unique_frames * i, :, :]
        for i in range(n_segments)
    ]

    for piece in pieces:
        assert piece.shape[0] == segment_frames

    new_feats = torch.cat(pieces, 0)
    return new_feats


@beartype
def get_segmenting_info(
    all_frames: int, overlap_frames: int, segment_frames: int
) -> Tuple[int, int]:
    """Get information of how to split the audio features

    This function calculates the padding that has to be concatenated
    at end of the original audio features, as well as the number of
    segments that the audio features will be split into. The segments need
    to have the duration corresponding to the sr_segment_frames number, and
    this includes the audio features overlap by sr_overlap_frames. The padding
    is added so that when the audio features are split in n_segments segments
    have the same dimension in order to make staking possible.

    Parameters
    ----------
    all_frames
        the initial number of frames in audio features
    overlap_frames
        the number of frames in a segment that will be copied from the previous segment
    segment_frames
        the number frames corresponding to the segment duration

    Returns
    -------
    n_segments
        how many segments of specific duration will exist
    padding
        how much to pad with zeros at the end
    """
    n_segments = (all_frames - overlap_frames) // (segment_frames - overlap_frames)
    remaining = (all_frames - overlap_frames) % (segment_frames - overlap_frames)
    assert (
        overlap_frames + n_segments * (segment_frames - overlap_frames) + remaining
        == all_frames
    ), "Something went wrong with segmentation, please check your arguments"

    if remaining == 0:
        padding = 0
    else:
        n_segments += 1
        padding = segment_frames - overlap_frames - remaining
    return n_segments, padding
