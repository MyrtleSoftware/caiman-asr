import random
from dataclasses import dataclass

from beartype import beartype
from beartype.typing import List, Optional

from caiman_asr_train.latency.client import ServerResponse
from caiman_asr_train.latency.client import fuse_timestamps as client_fuse
from caiman_asr_train.latency.client import get_word_timestamps
from caiman_asr_train.latency.timestamp import Never, group_timestamps
from caiman_asr_train.rnnt.response import (
    DecodingResponse,
    FrameResponses,
    HypothesisResponse,
)
from caiman_asr_train.utils.iter import flat
from caiman_asr_train.utils.responses import fuse_partials as ml_fuse
from caiman_asr_train.utils.responses import split_finals

TOKENS = [
    " ",
    "a",
    "b",
    "ab",
]


IDX_2_TOK = {idx: token for idx, token in enumerate(TOKENS)}

_TOK_IDXS = list(IDX_2_TOK.keys())


@beartype
@dataclass
class MiniHyp:
    tok: int
    time: int


@beartype
def rand_token() -> int:
    return random.choice(_TOK_IDXS)


@beartype
def simulate_server_response() -> dict[int, FrameResponses]:
    """
    Simulate beams progressing through a sentence
    emitting partials/finals at different timestamps
    """

    responses = {}

    step = 0
    max_steps = random.randint(0, 1000)

    acc = []

    while step < max_steps:
        # Append some tokens to the partial at this timestep
        for _ in range(random.randint(0, 2)):
            acc.append(MiniHyp(rand_token(), step))

        # With low probability, change the partial
        if random.random() < 0.1:
            for i in range(len(acc)):
                if random.random() < 0.25:
                    acc[i] = MiniHyp(rand_token(), acc[i].time)

        if random.random() < 0.2:
            # With low probability, emit a partial+final
            n = random.randint(0, len(acc))
            responses[step], acc = emit(acc[:], acc[n:]), acc[n:]
        else:
            # Otherwise, just emit the partial
            responses[step] = emit(acc[:], None)

        step += 1

    assert step not in responses

    responses[step] = emit(None, acc)

    return responses


@beartype
def emit(
    p_seq: Optional[List[MiniHyp]], f_seq: Optional[List[MiniHyp]]
) -> FrameResponses:
    partial = None if p_seq is None else emit_dec(False, p_seq)
    final = None if f_seq is None else emit_dec(True, f_seq)

    return FrameResponses(partials=partial, final=final)


@beartype
def emit_dec(is_final: bool, seq: List[MiniHyp]) -> DecodingResponse:
    """
    Transform a list of MiniHyp into a FrameResponses
    """
    return DecodingResponse(
        start_frame_idx=-1,
        duration_frames=-1,
        is_provisional=not is_final,
        alternatives=[
            HypothesisResponse(
                y_seq=[h.tok for h in seq],
                timesteps=[h.time for h in seq],
                token_seq=[IDX_2_TOK[h.tok] for h in seq],
                confidence=[1.0 for _ in seq],
            )
        ],
    )


@beartype
def convert_to_server(responses: dict[int, FrameResponses]) -> list[ServerResponse]:
    server_responses = []

    for frame, rsp in responses.items():
        if rsp.final is not None:
            server_responses.append(
                ServerResponse(
                    text="".join(rsp.final.alternatives[0].token_seq),
                    timestamp=float(frame),
                    is_partial=False,
                )
            )

        if rsp.partials is not None:
            server_responses.append(
                ServerResponse(
                    text="".join(rsp.partials.alternatives[0].token_seq),
                    timestamp=float(frame),
                    is_partial=True,
                )
            )

    return server_responses


def pretty(k, rsp: FrameResponses):
    if rsp.partials is not None:
        print(
            k,
            "Partial",
            rsp.partials.alternatives[0].token_seq,
            rsp.partials.alternatives[0].timesteps,
        )

    if rsp.final is not None:
        print(
            k,
            "Final",
            rsp.final.alternatives[0].token_seq,
            rsp.final.alternatives[0].timesteps,
        )


def test_vs_client():
    for index in range(100):
        print("\nIteration", index)

        responses = simulate_server_response()

        print("\nSimulated responses:")
        for k, v in responses.items():
            pretty(k, v)

        s_responses = convert_to_server(responses)

        print("\nServer responses:")
        for r in s_responses:
            print(r)

        out = client_fuse(s_responses)

        print("\nServer fused timestamps:")
        for ch, ts in out:
            print(ch, ts)

        ml_fused = ml_fuse(responses)

        copy_ml_fused = ml_fused

        print("\nML fused timestamps:")
        for k, v in responses.items():
            pretty(k, v)

        hyps = [
            f.final.alternatives[0] for f in ml_fused.values() if f.final is not None
        ]

        ml_toks = flat(h.token_seq for h in hyps)
        ml_tims = flat(h.timesteps for h in hyps)

        ml_fused = list(zip(ml_toks, ml_tims))

        out = []

        # Split joint tokens
        for chs, ts in ml_fused:
            for ch in chs:
                out.append((ch, ts))

        ml_fused = out

        print("\nML fused finals:")
        for ch, ts in ml_fused:
            print(ch, ts)

        for (ch1, ts1), (ch2, ts2) in zip(out, ml_fused):
            assert ch1 == ch2
            assert abs(ts1 - ts2) < 0.5

        # === Test word timestamps ===

        ml_tokens_int, ml_timestamps, _ = split_finals(copy_ml_fused)
        ml_tokens_str = [IDX_2_TOK[x] for x in ml_tokens_int]
        ml_sentence = "".join(ml_tokens_str)
        sequence_timestamp = group_timestamps(
            [ml_tokens_str], [ml_timestamps], [ml_sentence], [Never()]
        )[0]
        word_timestamps = [x.end_frame for x in sequence_timestamp.seqs]

        ##

        server_words_and_timestamps = get_word_timestamps(s_responses)

        if not word_timestamps:
            assert not server_words_and_timestamps
            continue

        _, server_timestamps = zip(*server_words_and_timestamps)

        if server_timestamps != word_timestamps:
            print("\nWord timestamps:")
            print(word_timestamps)
            print("\nServer word timestamps:")
            print(server_timestamps)
            # assert False

        assert (
            list(server_timestamps) == word_timestamps
        ), f"On iteration {index}, {list(server_timestamps)} != {word_timestamps}"
