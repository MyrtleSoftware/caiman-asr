from beartype import beartype
from beartype.typing import Dict, List, Tuple

from caiman_asr_train.rnnt.response import (
    DecodingResponse,
    FrameResponses,
    HypothesisResponse,
)
from caiman_asr_train.utils.iter import flat, repeat_like


@beartype
def split_finals(
    responses: Dict[int, FrameResponses],
) -> Tuple[List[int], List[int], List[float]]:
    """
    Convert a response to flat lists of y_seqs, timesteps and probabilities.
    """
    finals = [x.final for x in responses.values() if x.final]

    y_seqs = flat((final.alternatives[0].y_seq for final in finals))
    timesteps = flat(final.alternatives[0].timesteps for final in finals)
    probs = flat(final.alternatives[0].confidence for final in finals)

    return y_seqs, timesteps, probs


@beartype
def split_batched_finals(
    responses: List[Dict[int, FrameResponses]],
) -> Tuple[List[List[int]], List[List[int]], List[List[float]]]:
    """
    Apply split_finals to a batch of responses.
    """
    return tuple(map(list, zip(*[split_finals(r) for r in responses])))


@beartype
def fuse_partials(responses: Dict[int, FrameResponses]) -> Dict[int, FrameResponses]:
    """
    This function removes partial responses and modifies the timestamps of the
    finals to match the partials timestamps iff the partials match the final.


    Abstract example of algorithm:

    Partials:
        a b c d e
        a b c 1 2 3

    Finals:
        a b c 1 e

    Hence:
        a -> timestamp of first partial
        b -> timestamp of first partial
        c -> timestamp of first partial
        1 -> timestamp of second partial
        e -> timestamp of final

    Then:
        Carry 3 to next partial batch
    """

    new_responses = {}
    # The top partial from every response
    partials = []

    for response_idx, response in responses.items():
        # If the beam decoder produces a final and a partial on the
        # same timestep, the partial is for the following final.

        if (final := response.final) is not None:
            assert len(final.alternatives) == 1
            hyp = final.alternatives[0]

            assert len(hyp.y_seq) == len(hyp.timesteps) == len(hyp.token_seq)
            final_chars = flat(hyp.token_seq)

            # Worst case - assume latency of the final
            new_char_timesteps = [response_idx for _ in range(len(final_chars))]

            for final_char_idx, final_char in enumerate(final_chars):
                for partial_chars, partial_idx in reversed(partials):
                    if final_char_idx > len(partial_chars) - 1:
                        # This partial was short and wouldn't have overwritten
                        # final_char on the screen if final_char was already there.
                        # So keep rewinding time
                        continue
                    elif partial_chars[final_char_idx] == final_char:
                        # This partial matches the final, so the user
                        # would have seen this character continuously
                        # from now until the final
                        new_char_timesteps[final_char_idx] = partial_idx
                    else:
                        # Top partial doesn't match the final,
                        # so any prior matching partials don't count
                        # because they get overwritten by this non-matching partial
                        break

            # Map the worst char timesteps to token timesteps
            tok_idxs = repeat_like(range(len(hyp.y_seq)), _as=hyp.token_seq)

            new_timesteps = {}

            for char_idx, tok_idx in zip(new_char_timesteps, tok_idxs):
                new_timesteps[tok_idx] = max(char_idx, new_timesteps.get(tok_idx, 0))

            assert len(new_timesteps) == len(hyp.y_seq)

            new_timesteps = [new_timesteps[i] for i in range(len(hyp.y_seq))]

            # Build new final
            new_responses[response_idx] = FrameResponses(
                partials=None,
                final=DecodingResponse(
                    start_frame_idx=final.start_frame_idx,
                    duration_frames=final.duration_frames,
                    is_provisional=final.is_provisional,
                    alternatives=[
                        HypothesisResponse(
                            y_seq=hyp.y_seq,
                            timesteps=new_timesteps,
                            token_seq=hyp.token_seq,
                            confidence=hyp.confidence,
                        )
                    ],
                ),
            )

            # Carry over partials
            new_partials = []

            for partial_chars, partial_idx in partials:
                if len(partial_chars) > (s := len(final_chars)):
                    new_partials.append((partial_chars[s:], partial_idx))

            partials = new_partials
        else:
            new_responses[response_idx] = FrameResponses(partials=None, final=None)

        if (part := response.partials) is not None and part.alternatives:
            # Only the first is printed
            best = part.alternatives[0]

            assert all(t <= response_idx for t in best.timesteps), (
                response_idx,
                best.timesteps,
            )

            # A list of the individual characters from the partial:
            chars = flat(best.token_seq)
            partials.append((chars, response_idx))

    return new_responses
