# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, Myrtle Software Limited. All rights reserved.
# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Being an amalgamation of MLCommons MLPerf and NVIDIA Nemo code,

import os
from dataclasses import dataclass

import kenlm
import numpy as np
import torch
import torch.multiprocessing.spawn
from beartype import beartype
from beartype.typing import Dict, Generator, List, Optional, Tuple, Union, ValuesView
from sentencepiece import SentencePieceProcessor as SPP

from caiman_asr_train.keywords.process import load_keywords
from caiman_asr_train.keywords.trie import Keywords
from caiman_asr_train.lm.kenlm_ngram import KenLmModel, NgramInfo
from caiman_asr_train.rnnt.decoder import RNNTCommonDecoder
from caiman_asr_train.rnnt.eos_strategy import EOSPredict, EOSStrategy
from caiman_asr_train.rnnt.hypothesis import SPU_UNICODE, Hypothesis, init_sos_hyp
from caiman_asr_train.rnnt.model import RNNT
from caiman_asr_train.rnnt.response import FrameResponses
from caiman_asr_train.rnnt.serialise_responses import ResponseSerializer


@beartype
@dataclass
class Work:
    """
    A packet of encoding work to be sent to the GPU.
    """

    hyp: Hypothesis
    enc: torch.Tensor


@beartype
@dataclass
class Decoding:
    """
    The result of beam decoding an utterance.
    """

    responses: Dict[int, FrameResponses]


@beartype
@dataclass
class GThread:
    """
    Manages a generator decoding an utterance, the index is the
    index of the utterance in the beam-wide batch.
    """

    idx: int
    gen: Generator


@beartype
class RNNTBeamDecoder(RNNTCommonDecoder):
    """A beam transducer decoder.

    Args:
        model: RNN-T model
        blank_idx: Index of blank token - assumed to be at the end of the vocab
        sentpiece_model: Tokenizer's sentence piece model.
        beam_width : Width of the search beam - an integer > 0
        max_inputs_per_batch: Max number of tensor elements per batch - default 10 million.
            See `docs/src/training/automatic_batch_size_reduction.md` for more details.
        max_symbols_per_step: The maximum number of symbols that can be
            decoded per timestep; if set to None then there is no limit - default 8.
        max_symbol_per_sample: The maximum number of symbols that can be
            decoded per utterance; if set to None (default) then there is no limit.
        temperature: Temperature parameter for scaling logits - default 1.4.
        beam_prune_score_thresh: Hypotheses with a score per token that is
            `beam_prune_score_thresh` less than the best hypothesis' score will be pruned.
        beam_prune_topk_thresh: Tokens with a log_prob score `beam_prune_topk_thresh`
            less than the most likely token will not be considered.
        ngram_info: NgramInfo dataclass containing file path to n-gram language model
            and weight of n-gram score.
        fuzzy_topk_logits: Reduce the logits tensor to match the tensor that the solution
            is based on.
        return_partials: Return partial hypotheses in the output (this has a non-zero
            performance cost), if this is false then some responses may have neither a
            partial nor a final response.
        user_tokens: List of meta tokens that should not be corrected by the n-gram model.
        eos_is_terminal: If True, the EOS token is treated as a terminal token and the
            search will terminate when it is predicted. If False, the search will continue.
        eos_vad_threshold: If this many seconds of silence are detected, the search will
            terminate early.
        final_emission_thresh: If the time between final emissions exceeds this value,
            the search will discard partial hypotheses until a final emission is made.
        frame_width: The width of the encoder output frames in seconds. This is only
            required if `eos_vad_threshold` or `final_emission_thresh` is set to a
            non-infinite value.
    """

    def __init__(
        self,
        model: RNNT,
        blank_idx: int,
        eos_strategy: EOSStrategy,
        sentpiece_model: str,
        beam_width: int = 4,
        max_inputs_per_batch: int = int(1e7),
        max_symbols_per_step: Optional[int] = 8,
        max_symbol_per_sample: Optional[int] = None,
        temperature: float = 1.4,
        beam_prune_score_thresh: Union[int, float] = 0.4,
        beam_prune_topk_thresh: Union[int, float] = 1.5,
        ngram_info: Optional[NgramInfo] = None,
        fuzzy_topk_logits: bool = False,
        return_partials: bool = False,
        user_tokens: Optional[List[int]] = None,
        eos_is_terminal: bool = False,
        eos_vad_threshold: float = float("inf"),
        final_emission_thresh: float = float("inf"),
        frame_width: Optional[float] = None,
        keyword_boost_path: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            blank_idx=blank_idx,
            eos_strategy=eos_strategy,
            max_inputs_per_batch=max_inputs_per_batch,
            max_symbol_per_sample=max_symbol_per_sample,
            max_symbols_per_step=max_symbols_per_step,
            temperature=temperature,
        )
        assert beam_width > 0
        self.beam_width = beam_width
        self.normalised_score = lambda x: x.score / x.y_length_tot

        inf = float("inf")

        if final_emission_thresh < 0:
            final_emission_thresh = inf

        if eos_vad_threshold != inf or final_emission_thresh != inf:
            assert frame_width is not None
            assert frame_width > 0.0

        self.eos_vad_threshold = eos_vad_threshold
        self.final_emission_thresh = final_emission_thresh
        self.frame_width = 0.0 if frame_width is None else frame_width

        self.detokenize = SPP(model_file=sentpiece_model).id_to_piece
        self.spu_unicode = SPU_UNICODE
        self.user_tokens = [] if user_tokens is None else user_tokens
        self.eos_is_terminal = eos_is_terminal

        if keyword_boost_path is None:
            self.keywords = Keywords[str]([])
        else:
            self.keywords = load_keywords(keyword_boost_path)

        if ngram_info:
            ngram_path, self.ngram_alpha = ngram_info.path, ngram_info.scale_factor
            assert os.path.isfile(
                ngram_path
            ), f"N-gram LM path {ngram_path} does not exist."
            assert (
                self.ngram_alpha >= 0.0
            ), f"N-gram scale factor is negative, {self.ngram_alpha}"
            self.ngram_lm = KenLmModel(ngram_path)
        else:
            self.ngram_lm = None

        self.beam_prune_topk_thresh = beam_prune_topk_thresh
        self.beam_prune_score_thresh = beam_prune_score_thresh
        self.fuzzy_topk_logits = fuzzy_topk_logits

        if self.beam_prune_topk_thresh < 0:
            self.beam_prune_topk_thresh = float("inf")

        assert self.beam_prune_topk_thresh > 1e-9, (
            "--beam_prune_topk_thresh=0 means that all tokens are pruned except "
            "for the most probable. If this is the desired behaviour consider running "
            "`--decoder=greedy` instead."
        )

        if self.beam_prune_score_thresh < 0:
            self.beam_prune_score_thresh = float("inf")

        assert self.beam_prune_score_thresh > 1e-9, (
            "--beam_prune_score_thresh=0 means that all hypotheses are pruned except "
            "for the most probable. If this is the desired behaviour consider running "
            "`--decoder=greedy` instead."
        )

        self.serialiser = ResponseSerializer(self._sort_nbest)
        self.return_partials = return_partials

    @torch.no_grad()
    def _inner_decode(
        self, encs: torch.Tensor, encs_len: torch.Tensor
    ) -> List[Dict[int, FrameResponses]]:
        training_state = self.model.training
        #
        self.model.eval()

        B, _, _ = encs.shape

        thr = [
            GThread(i, self._gloop(encs[None, i], encs_len[None, i])) for i in range(B)
        ]

        pend = [(t, t.gen.send(None)) for t in thr]
        done = {}

        while pend:
            y_pred = []
            n_pred = []

            for t, res in pend:
                match res:
                    case Decoding(responses):
                        done[t.idx] = responses

                    case Work(hyp, enc):
                        if hyp.pred_state is None:
                            n_pred.append((t, hyp, enc))
                        else:
                            y_pred.append((t, hyp, enc))

                    case _:
                        raise ValueError("Unexpected result")

            pend = []

            def ingest(batch, has_pred_state):
                if not batch:
                    return

                outs = self._batched_decode_step(encs, batch, has_pred_state)

                pend.extend(
                    (t, t.gen.send(pack)) for (t, _, _), pack in zip(batch, outs)
                )

            ingest(n_pred, False)
            ingest(y_pred, True)

        self.model.train(training_state)

        return [r for _, r in sorted(done.items())]

    @beartype
    def _silence_terminate(self, kept_hyps: dict[int, Hypothesis], idx: int) -> bool:
        """
        Decide if the decoding should stop
        """
        if self.eos_vad_threshold == float("inf"):
            return False

        # Find the most recent non-blank token
        last_step = max(hyp.timesteps[-1] for hyp in kept_hyps.values())

        if last_step < 0:
            # The SOS token is emitted at timestamp -1, If all the
            # hypotheses are empty we should not terminate.
            return False

        silence = (idx - last_step) * self.frame_width

        return silence >= self.eos_vad_threshold

    def _gloop(self, encs: torch.Tensor, encs_len: torch.Tensor) -> Generator:
        """
        Run decoding loop given encoder features.

        encs is shape (1, time, enc_dim)
        """

        blank_tensor = torch.tensor([self.blank_idx], device="cpu", dtype=torch.long)

        sos_hyp = init_sos_hyp(self._SOS, self.ngram_lm)

        kept_hyps = {sos_hyp.hashval: sos_hyp}

        responses: Dict[int, FrameResponses] = {}

        last_final_idx = 0

        for time_idx in range(encs_len.max()):
            # Stop if the length of the best hypothesis exceeds max_symbol_per_sample
            if self.max_symbol_per_sample is not None:
                max_hyp = max(kept_hyps.values(), key=lambda x: x.score)
                if max_hyp.y_length_tot > self.max_symbol_per_sample:
                    break

            # encs_t is the acoustic encoding for this timestep, shape (batch, 1, enc_dim)
            encs_t = encs[:, None, time_idx, :]

            kept_hyps = yield from self._beam_run_timestep(
                encs_t, kept_hyps, time_idx, blank_tensor
            )

            if max(kept_hyps.values(), key=lambda x: x.score).is_terminal:
                # Early stop
                responses[time_idx] = self.serialiser.last_frame_response(kept_hyps)
                yield Decoding(responses)
                return

            time_since_final = (time_idx - last_final_idx) * self.frame_width

            while True:
                responses[time_idx], kept_hyps = self.serialiser.frame_responses(
                    kept_hyps, time_idx, self.return_partials
                )

                if len(kept_hyps) <= 1:
                    # This can happen if the partial contains no new tokens, hence
                    # partial is same as final modulo blank, hence bump idx.
                    last_final_idx = time_idx
                    break

                if responses[time_idx].final is not None:
                    # This means final was emitted but still more than one hypothesis
                    # Hence we use the oldest partial as the common ancestor time.

                    last_final_idx = min(hyp.timesteps[0] for hyp in kept_hyps.values())
                    break

                if time_since_final <= self.final_emission_thresh:
                    break

                # Pruning the least probable hypothesis until a final is emitted
                kept_hyps.pop(
                    min(kept_hyps.values(), key=self.normalised_score).hashval
                )

            if self._silence_terminate(kept_hyps, time_idx):
                break

        responses[time_idx + 1] = self.serialiser.last_frame_response(kept_hyps)
        yield Decoding(responses)

    def _beam_run_timestep(
        self,
        enc_t: torch.Tensor,
        hyps: Dict[int, Hypothesis],
        time_idx: int,
        blank_tensor: torch.Tensor,
    ) -> Generator:
        """
        Run a single timestep of the beam search.
        """
        kept_hyps = dict()  # Kept hyps must end in a blank on this timestep

        for hyp in hyps.values():
            hyp.y_len_t = 0

        while hyps:
            # Get best hypothesis
            max_hyp_hashval = max(hyps.values(), key=lambda x: x.score).hashval
            max_hyp = hyps.pop(max_hyp_hashval)

            top_k_score, top_k_idx, blank_p, pred_state = yield Work(max_hyp, enc_t)

            steps = self._prepare_steps(
                top_k_score, top_k_idx, blank_p, blank_tensor, self.add_ys(max_hyp)
            )

            for klog_p, kidx in zip(*steps):
                hyps, kept_hyps = self._update_hyps(
                    klog_p,
                    kidx,
                    max_hyp,
                    kept_hyps,
                    hyps,
                    time_idx,
                    pred_state,
                )

            # Continue until at least beam_width hypotheses in kept are better than
            # the best in `hyps`. With a small max_symbols, `hyps` may be empty before this.
            # If so, feed the best beam_width hypotheses in kept into the next timestep.

            if hyps:
                max_hyp_score = max(hyps.values(), key=lambda x: x.score).score
                kept_better = {
                    has: hyp
                    for (has, hyp) in kept_hyps.items()
                    if hyp.score > max_hyp_score
                }
                # If enough hypotheses have better scores than
                # the highest scoring hypothesis in `hyps`
                if len(kept_better) >= self.beam_width:
                    # Feed the 'better' hyps into the next timestep
                    kept_hyps = self._best_beam_width(kept_better)
                    break
            else:
                # Feed the best hypotheses into the next timestep
                kept_hyps = self._best_beam_width(kept_hyps)
                break

        return self._prune_beam(kept_hyps)

    def add_ys(self, hyp: Hypothesis) -> bool:
        """
        Determine whether a hypothesis should be expanded with non-blank tokens.
        """
        return not self.max_symbols or (hyp.y_len_t < self.max_symbols)

    def _prepare_steps(
        self, top_k_score, top_k_idx, blank_p, blank_tensor: torch.Tensor, add_ys: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare hypothesis expansion steps.
        """

        steps = (top_k_score, top_k_idx)

        if add_ys:
            # Add blank if not present in the top_k. This is rare for a trained
            # model at large beam sizes, but can occur with a small beam size.
            if self.blank_idx not in steps[1]:
                steps = (
                    torch.cat((steps[0], blank_p.unsqueeze(0))),
                    torch.cat((steps[1], blank_tensor)),
                )
        else:
            # Only prepare the blank
            steps = (
                blank_p.unsqueeze(0).cpu(),
                blank_tensor.cpu(),
            )

        return steps

    def _update_hyps(
        self,
        klog_p: torch.Tensor,
        kidx: torch.Tensor,
        max_hyp: Hypothesis,
        kept_hyps: Dict[int, Hypothesis],
        hyps: Dict[int, Hypothesis],
        time_idx: int,
        pred_state_prime: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[Dict[int, Hypothesis], Dict[int, Hypothesis]]:
        """
        Create a new hypothesis and update the beam

        Hypothesis merging is done by checking the hash of the new hypothesis against those
        in the kept_hyps/hyps dicts.
        """

        if kidx.item() == self.blank_idx:
            if max_hyp.hashval in kept_hyps:
                kept_hyps[max_hyp.hashval].score = np.logaddexp(
                    kept_hyps[max_hyp.hashval].score, max_hyp.score + float(klog_p)
                )
            else:
                new_hyp = max_hyp.clone()
                new_hyp.score += float(klog_p)
                kept_hyps[new_hyp.hashval] = new_hyp

            return hyps, kept_hyps

        new_hyp = max_hyp.clone()
        new_hyp.score += float(klog_p)
        new_hyp.p_seq.append(float(klog_p.exp()))

        new_hyp.timesteps.append(time_idx)
        new_hyp.pred_state = pred_state_prime
        new_hyp.y_seq.append(int(kidx))
        new_hyp.y_len_t += 1

        if self.eos_is_terminal:
            match self.eos_strategy:
                case EOSPredict(idx, _, _):
                    if kidx.item() == idx:
                        new_hyp.is_terminal = True

        if kidx.item() not in self.user_tokens:
            # Skip n-gram correction for meta tokens which the
            # n-gram model is not trained on.
            new_hyp = self._ngram_correction(max_hyp.ngram_lm_state, new_hyp)

        new_hyp = self._keyword_correction(new_hyp)

        token_str, cleaned_str = self._get_token_str(new_hyp, kidx)
        new_hyp.s_seq.append(token_str)

        if cleaned_str:
            # If an underscore was followed by an underscore don't update hash
            new_hyp.update_hash(cleaned_str)

        if new_hyp.hashval in hyps:
            summed_score = np.logaddexp(hyps[new_hyp.hashval].score, new_hyp.score)
            # Keep more probable y sequence, and hence states when merging hyps
            if new_hyp.score > hyps[new_hyp.hashval].score:
                hyps[new_hyp.hashval] = new_hyp
            hyps[new_hyp.hashval].score = summed_score
        else:
            hyps[new_hyp.hashval] = new_hyp

        return hyps, kept_hyps

    def _batched_decode_step(
        self,
        encs: torch.Tensor,
        batch: List[Tuple[GThread, Hypothesis, torch.Tensor]],
        has_pred_state: bool,
    ) -> List[
        Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
        ]
    ]:
        """
        Run predict and joint steps for a batch of hypotheses and
        package the results into packets for each generator.
        """

        log_p, h, c = self._batched_decode(encs, batch, has_pred_state)

        top_k_score, top_k_idx = log_p.topk(self.beam_width, dim=1)
        max_score, _ = top_k_score.max(dim=1, keepdim=True)
        keep = top_k_score >= max_score - self.beam_prune_topk_thresh

        # Compute offsets post boolean indexing. We will need offsets on the CPU
        # for the slicing in the next step so we avoid the copy here.
        offsets = torch.zeros(keep.shape[0] + 1, dtype=torch.long, device="cpu")
        tmp = keep.sum(dim=1).cumsum(dim=0)
        offsets[1:] = tmp

        top_k_score, top_k_idx = top_k_score[keep].cpu(), top_k_idx[keep].cpu()
        blank_p = log_p[:, self.blank_idx].cpu().unbind()

        hs = h.unsqueeze(2).unbind(1)
        cs = c.unsqueeze(2).unbind(1)

        # Process into packets

        packets = []

        if has_pred_state:
            for h, c, ps, lo, hi in zip(hs, cs, blank_p, offsets[:-1], offsets[1:]):
                packets.append((top_k_score[lo:hi], top_k_idx[lo:hi], ps, (h, c)))
        else:
            for ps, lo, hi in zip(blank_p, offsets[:-1], offsets[1:]):
                packets.append((top_k_score[lo:hi], top_k_idx[lo:hi], ps, (h, c)))

        return packets

    def _batched_decode(
        self,
        encs: torch.Tensor,
        batch: List[Tuple[GThread, Hypothesis, torch.Tensor]],
        has_pred_state: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run predict and joint steps for a batch of hypotheses.
        """
        f, y, state = self._collate(encs.device, batch, has_pred_state)

        g, (h, c), _ = self._pred_step_raw(y, state)

        log_p = self._joint_step(
            f,
            g,
            fuzzy=self.fuzzy_topk_logits,
        )

        return log_p, h, c

    def _collate(
        self,
        device,
        batch: List[Tuple[GThread, Hypothesis, torch.Tensor]],
        has_pred_state: bool,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        Stitch together y_last values and hidden states for batched evaluation.
        """
        f = torch.cat([enc for _, _, enc in batch], dim=0)

        if has_pred_state:
            # Collate y_last values (shape (B, U=1))
            y = torch.tensor(
                [[h.y_last] for _, h, _ in batch], dtype=torch.long, device=device
            )

            # Collate hidden states (shape (L, B, H))
            h = torch.cat([s.pred_state[0] for _, s, _ in batch], dim=1)
            c = torch.cat([s.pred_state[1] for _, s, _ in batch], dim=1)

            return f, y, (h, c)

        return f, None, None

    def _keyword_correction(self, new_hyp: Hypothesis) -> Hypothesis:
        """
        Apply the keyword correction to the new hypothesis.
        """
        if self.keywords is None:
            return new_hyp

        assert new_hyp.y_last != 0, "Decoding error: '<unk>' token encountered"

        new_token = self.detokenize(new_hyp.y_last)
        delta, new_hyp.kws_state = self.keywords.steps(new_token, new_hyp.kws_state)
        new_hyp.score += delta

        return new_hyp

    def _ngram_correction(
        self, max_hyp_ngram_state: Optional[kenlm.State], new_hyp: Hypothesis
    ) -> Hypothesis:
        """Apply n-gram language model correction to the new hypothesis."""
        if not self.ngram_lm:
            return new_hyp
        assert new_hyp.y_last != 0, "Decoding error: '<unk>' token encountered"
        new_token = self.detokenize(new_hyp.y_last)
        lm_score, new_hyp.ngram_lm_state = self.ngram_lm.score_ngram(
            new_token,
            max_hyp_ngram_state,
        )
        new_hyp.score += self.ngram_alpha * lm_score
        return new_hyp

    def _get_token_str(
        self, new_hyp: Hypothesis, kidx: torch.Tensor
    ) -> Tuple[str, str]:
        """
        Return tuple of token string and cleaned token string.

        Cleaned token string skips any new leading underscores.
        """
        token_str = self.detokenize(int(kidx))

        s_last = new_hyp.s_seq[-1][-1]  # Final character of final string in s_seq
        return token_str, (
            token_str[1:]
            if ord(s_last) == ord(token_str[0]) == self.spu_unicode
            else token_str
        )

    def _best_beam_width(
        self, kept_hyps: Dict[int, Hypothesis]
    ) -> Dict[int, Hypothesis]:
        """
        Return the best self.beam_width hypotheses from kept_hyps.
        """
        # If there are fewer than beam_width hypotheses in kept_hyps then return them all
        if len(kept_hyps) <= self.beam_width:
            return kept_hyps
        # Otherwise return beam_width hypotheses with highest score
        kept_sorted = sorted(kept_hyps.values(), key=lambda x: x.score, reverse=True)
        return {hyp.hashval: hyp for hyp in kept_sorted[: self.beam_width]}

    def _prune_beam(self, hyps: Dict[int, Hypothesis]) -> Dict[int, Hypothesis]:
        """Prune hypotheses to keep those within the beam score threshold."""
        max_hyp_score = self.normalised_score(
            max(hyps.values(), key=self.normalised_score)
        )
        return {
            k: v
            for k, v in hyps.items()
            if self.normalised_score(v) >= max_hyp_score - self.beam_prune_score_thresh
        }

    def _sort_nbest(self, hyps: ValuesView[Hypothesis]) -> List[Hypothesis]:
        """Sort hypotheses by length normalized score."""
        return sorted(hyps, key=self.normalised_score, reverse=True)
