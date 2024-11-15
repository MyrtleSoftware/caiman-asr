from dataclasses import dataclass

from beartype import beartype


@beartype
@dataclass
class HypothesisResponse:
    """
    {
        "y_seq": [1, 2, 3],
        "timesteps": [0, 1, 4],
        "token_seq": ["▁hello", "▁wo", "rld"],
        "confidence": [1.0, 1.0, 1.0],
    }
    """

    y_seq: list[int]
    timesteps: list[int]
    token_seq: list[str]
    confidence: list[float]


@beartype
@dataclass
class DecodingResponse:
    """
    {
        "start_frame_idx": 0,
        "duration_frames": 2,
        "is_provisional": false,
        "alternatives": [
            ...
        ]
    }
    """

    start_frame_idx: int
    duration_frames: int
    is_provisional: bool
    alternatives: list[HypothesisResponse]


@beartype
@dataclass
class FrameResponses:
    """
    Decoding responses for a single frame of audio.

    Responses can be 'partial' or 'final' where:
    * Partial responses are hypotheses that are provisional and may be removed or
    updated in future frames
    * Final responses are hypotheses that are complete and will not change in future
    frames

    It is recommended to use partials for low-latency streaming applications and finals
    for the ultimate transcription output. If latency is not a concern you can ignore
    the partials and concatenate the finals.

    During beam search decoding:
    * Every frame (other than the last one) will have partials and may also have a final,
    unless the `--beam_no_partials` flag is set.
    * The last frame will not have partials and may have a final if there is outstanding
    text.

    During greedy decoding:
    * All responses are final and there are no partials.
    * If the network returns a blank token a `FrameResponses(None, None)` will be returned.

    """

    partials: DecodingResponse | None
    final: DecodingResponse | None
