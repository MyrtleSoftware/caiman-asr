from pydantic import BaseModel


class HypothesisResponse(BaseModel):
    """
    {
        "y_seq": [1, 2, 3],
        "timesteps": [0, 1, 4],
        "token_seq": ["▁hello", "▁wo", "rld"],
        "confidence": 1.0,
    }
    """

    y_seq: list[int]
    timesteps: list[int]
    token_seq: list[str]
    confidence: float


class DecodingResponse(BaseModel):
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


class FrameResponses(BaseModel):
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
    ANCHOR: finals_partials_in_mdbook
    * Every frame other than the last one will have partials and may also have a final
    * The last frame will not have partials and may have a final if there is outstanding
    text.
    ANCHOR_END: finals_partials_in_mdbook
    """

    partials: DecodingResponse | None
    final: DecodingResponse | None
