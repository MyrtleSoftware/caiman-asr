from beartype.typing import Callable, Optional

from caiman_asr_train.rnnt.hypothesis import Hypothesis
from caiman_asr_train.rnnt.response import (
    DecodingResponse,
    FrameResponses,
    HypothesisResponse,
)


class ResponseSerializer:
    """
    Generate beam FrameResponses from hypotheses.

    See FrameResponses docstring for more information.

    A final is detected when all hypotheses share a common prefix - which guarantees
    highest accuracy of the finals at the expense of larger lag between the finals.
    Other less conservative final detection algorithms are possible.

    NOTE: this isn't a true serialiser in that it just creates a python FrameResponses
    class, but that object and its children are dataclasses that can be
    serialised to JSON with the dataclasses.asdict() method.
    """

    def __init__(self, nbest_sort: Callable) -> None:
        self.nbest_sort = nbest_sort

    def frame_responses(
        self,
        kept_hyps: dict[int, Hypothesis],
        time_idx: Optional[int] = None,
        partials=True,
    ) -> tuple[FrameResponses, dict[int, Hypothesis]]:
        """
        Build the frame-level responses and update kept_hyps if necessary.

        kept_hyps is only updated if a final is selected.

        If partials is True then partials will be built, otherwise they will be None.

        Note: building partials is very slow and currently we don't use them so
        myrtle has not invested resources in optimizing this option.
        """
        final, kept_hyps = self._get_final(kept_hyps)

        if partials:
            assert time_idx is not None, "time_idx must be provided if partials is True"
            partials = self._build_partials(kept_hyps, time_idx)
        else:
            partials = None

        return FrameResponses(partials=partials, final=final), kept_hyps

    def last_frame_response(
        self,
        kept_hyps: dict[int, Hypothesis],
    ) -> FrameResponses:
        """
        Build the frame-level response for the closing frame.

        In this case, any remaining transcript in the top-beam hypothesis is sent as
        the final and there are no partials.

        This response is sent one timestep after the last audio frame.
        """
        best_hyp = self.nbest_sort(kept_hyps.values())[0]

        final = None
        if len(best_hyp.y_seq) > 1:
            final = self._build_final([best_hyp], len(best_hyp.y_seq))

        return FrameResponses(partials=None, final=final)

    def _build_partials(
        self, kept_hyps: dict[int, Hypothesis], time_idx: int
    ) -> DecodingResponse:
        """
        Build the partial DecodingResponse.

        All hypotheses in the beam will be returned as alternatives in the partial
        DecodingResponse.
        """
        n_best_list = self.nbest_sort(kept_hyps.values())

        alternatives = []
        start_frame = time_idx
        for hyp in n_best_list:
            timesteps, token_seq, y_seq, p_seq = (
                hyp.timesteps[1:],
                hyp.s_seq[1:],
                hyp.y_seq[1:],
                hyp.p_seq[1:],
            )
            assert len(timesteps) == len(token_seq) == len(y_seq)
            if not timesteps:
                # don't serialize the hypothesis
                continue
            start_frame = min(start_frame, min(timesteps))
            alt = HypothesisResponse(
                y_seq=y_seq, timesteps=timesteps, token_seq=token_seq, confidence=p_seq
            )
            alternatives.append(alt)

        return DecodingResponse(
            start_frame_idx=start_frame,
            duration_frames=time_idx - start_frame + 1,
            is_provisional=True,
            alternatives=alternatives,
        )

    def _get_final(
        self, kept_hyps: dict[int, Hypothesis]
    ) -> tuple[DecodingResponse | None, dict[int, Hypothesis]]:
        """
        Select the final if present and build the final DecodingResponse.

        The selection of a final is done by checking if all hypotheses share a common
        prefix as in this case it impossible for the transcription before this point to
        change.

        If a final is selected then the shared prefix is removed from all hypotheses.
        """
        # find common token prefix between the first and last after sorting alphabetically
        hyps_ls = sorted(kept_hyps.values(), key=lambda x: x.s_seq)
        first = hyps_ls[0].s_seq
        last = hyps_ls[-1].s_seq

        max_prefix_length = min(len(first), len(last))

        # ignore first token as this is either SOS or previously shipped as a final
        tkn_idx = 1
        while tkn_idx < max_prefix_length and first[tkn_idx] == last[tkn_idx]:
            tkn_idx += 1

        if tkn_idx == 1:  # no new final
            return None, kept_hyps

        final = self._build_final(hyps_ls, tkn_idx)

        # remove prefix from all hypotheses
        for hyp in kept_hyps.values():
            hyp.truncate(tkn_idx)

        return final, kept_hyps

    def _build_final(self, hyps: list[Hypothesis], tkn_idx: int) -> DecodingResponse:
        """
        Build the final DecodingResponse from a list of hyps.

        This method assumes that there is a shared prefix between all hypotheses up to
        tkn_idx.

        Args:
            hyp: hypothesis
            tkn_idx: token index up to which the final should be built from
        """
        s_seqs, y_seqs, timesteps, p_seqs = [], [], [], []
        for hyp in hyps:
            p_seqs.append(hyp.p_seq[1:tkn_idx])
            s_seqs.append(hyp.s_seq[1:tkn_idx])
            y_seqs.append(hyp.y_seq[1:tkn_idx])
            timesteps.append(hyp.timesteps[1:tkn_idx])

        assert all(
            s_seqs[0] == s for s in s_seqs[1:]
        ), f"All s_seqs should be the same but {s_seqs=}"
        assert all(
            y_seqs[0] == y for y in y_seqs[1:]
        ), f"All y_seqs should be the same but {y_seqs=}"

        final_s_seq = s_seqs[0]
        final_y_seq = y_seqs[0]
        final_p_seq = p_seqs[0]

        # timesteps are not guaranteed to be the same across hypotheses so take the
        # minimum for each token. This is done because if the token was output by the
        # model with high probability then it has almost certainly been spoken by that
        # point
        assert all(
            len(t) == len(timesteps[0]) for t in timesteps[1:]
        ), f"All timesteps should be the same length but {timesteps=}"

        final_timesteps = [
            min(timestep[i] for timestep in timesteps) for i in range(len(timesteps[0]))
        ]

        start_frame, end_frame = min(final_timesteps), max(final_timesteps)

        final_response = HypothesisResponse(
            y_seq=final_y_seq,
            timesteps=final_timesteps,
            token_seq=final_s_seq,
            confidence=final_p_seq,
        )
        return DecodingResponse(
            start_frame_idx=start_frame,
            duration_frames=end_frame - start_frame + 1,
            is_provisional=False,
            alternatives=[final_response],
        )
