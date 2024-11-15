from enum import Enum

from caiman_asr_train.rnnt.response import (
    DecodingResponse,
    FrameResponses,
    HypothesisResponse,
)
from caiman_asr_train.utils.responses import fuse_partials


class Kind(Enum):
    PARTIAL = 0
    FINAL = 1


TOKEN_TO_STRING = ["<unk>", "a", "b", "c", "d", "e", "f"]


def mock_response(ys, ts, seq=None, kind: Kind = Kind.FINAL):
    assert len(ys) == len(ts)

    if seq is not None:
        assert len(ys) == len(seq)

    seq = [TOKEN_TO_STRING[y] for y in ys] if seq is None else seq

    rsp = DecodingResponse(
        start_frame_idx=-1,
        duration_frames=-1,
        is_provisional=kind is Kind.PARTIAL,
        alternatives=[
            HypothesisResponse(
                y_seq=ys,
                timesteps=ts,
                token_seq=seq,
                confidence=[1.0 for _ in ys],
            )
        ],
    )

    return FrameResponses(
        partials=rsp if kind is Kind.PARTIAL else None,
        final=None if kind is Kind.PARTIAL else rsp,
    )


def test_beam_fuse_responses():
    responses = {
        0: mock_response([1], [0], kind=Kind.PARTIAL),
        1: mock_response([1, 2], [0, 1], kind=Kind.PARTIAL),
        2: mock_response([1, 2, 3], [0, 1, 2], kind=Kind.PARTIAL),
        5: mock_response([1, 2, 4, 5], [0, 1, 5, 5], kind=Kind.PARTIAL),
        9: mock_response([1, 2, 4, 6], [0, 1, 2, 3], kind=Kind.FINAL),
    }

    responses = fuse_partials(responses)

    assert len(responses) == 5
    assert 9 in responses
    assert responses[9].final.alternatives[0].timesteps == [0, 1, 5, 9]


def test_beam_tok_swap():
    responses = {
        1: mock_response([0, 2], [0, 1], ["a", "b"], Kind.PARTIAL),
        2: mock_response([3], [2], ["ab"], Kind.FINAL),
    }

    responses = fuse_partials(responses)

    assert len(responses) == 2
    assert 2 in responses
    assert responses[2].final.alternatives[0].timesteps == [1]


def test_final_with_early_times():
    """Without the knowledge that token 3 = two of token 2,
    the timestep goes up to 4, even though the final claims
    it knew the token at time 2"""
    responses = {
        1: mock_response([0, 2], [0, 1], kind=Kind.PARTIAL),
        2: mock_response([2, 2, 4], [0, 1, 2], kind=Kind.PARTIAL),
        4: mock_response([3, 4], [2, 2], kind=Kind.FINAL),
    }

    responses = fuse_partials(responses)

    assert responses[4].final.alternatives[0].y_seq == [3, 4]
    assert responses[4].final.alternatives[0].timesteps == [4, 4]


def test_beam_tok_split():
    responses = {
        1: mock_response([0, 2], [0, 1], ["a", "b"], Kind.PARTIAL),
        2: mock_response([2, 2, 4], [0, 1, 2], ["b", "b", "c"], Kind.PARTIAL),
        4: mock_response([3, 4], [2, 2], ["bb", "c"], Kind.FINAL),
    }

    responses = fuse_partials(responses)

    assert len(responses) == 3
    assert 4 in responses
    assert responses[4].final.alternatives[0].timesteps == [2, 2]


def pretty(k, rsp: FrameResponses):
    if rsp.partials is not None:
        print(k, "Partial", rsp.partials.alternatives[0].token_seq)
    if rsp.final is not None:
        print(k, "Final", rsp.final.alternatives[0].token_seq)


def test_beam_carry():
    responses = {
        0: mock_response([1], [0], kind=Kind.PARTIAL),
        1: mock_response([1, 2, 3], [0, 1, 1], kind=Kind.PARTIAL),
        4: mock_response([1], [2], kind=Kind.FINAL),
        5: mock_response([2, 3], [3, 3], kind=Kind.FINAL),
    }

    for k, v in responses.items():
        pretty(k, v)

    responses = fuse_partials(responses)

    print()

    for k, v in responses.items():
        pretty(k, v)

    timestamps = (
        responses[4].final.alternatives[0].timesteps
        + responses[5].final.alternatives[0].timesteps
    )

    assert timestamps == [0, 1, 1]


def test_greedy_responses():
    N = 10

    responses = {i: mock_response([-1, -1], [i, i], kind=Kind.FINAL) for i in range(N)}

    responses = fuse_partials(responses)

    assert len(responses) == N

    for i in range(N):
        assert i in responses
        assert responses[i].final.alternatives[0].timesteps == [i, i]
