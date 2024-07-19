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

import kenlm
import numpy as np
import torch
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple, Union, ValuesView
from cachetools import FIFOCache

from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.lm.kenlm_ngram import KenLmModel, NgramInfo
from caiman_asr_train.rnnt.decoder import RNNTDecoder
from caiman_asr_train.rnnt.hypothesis import SPU_UNICODE, Hypothesis, init_sos_hyp
from caiman_asr_train.rnnt.model import RNNT
from caiman_asr_train.rnnt.response import FrameResponses
from caiman_asr_train.rnnt.serialise_responses import ResponseSerializer


@beartype
class RNNTBeamDecoder(RNNTDecoder):
    """A beam transducer decoder.

    Args:
        model: RNN-T model
        blank_idx: Index of blank token - assumed to be at the end of the vocab
        tokenizer: Sentencepiece tokenizer.
        beam_width : Width of the search beam - an integer > 0
        max_inputs_per_batch: Max number of tensor elements per batch - default 10 million.
            See `docs/src/training/automatic_batch_size_reduction.md` for more details.
        max_symbols_per_step: The maximum number of symbols that can be
            decoded per timestep; if set to None then there is no limit - default 8.
        max_symbol_per_sample: The maximum number of symbols that can be
            decoded per utterance; if set to None (default) then there is no limit.
        temperature: Temperature parameter for scaling logits - default 1.4.
        pred_net_cache_size: Size of prediction network cache - default 100.
        beam_prune_score_thresh: Hypotheses with a score per token that is
            `beam_prune_score_thresh` less than the best hypothesis' score will be pruned.
        beam_prune_topk_thresh: Tokens with a logprob score `beam_prune_topk_thresh`
            less than the most likely token will not be considered.
        ngram_info: NgramInfo dataclass containing file path to n-gram language model
            and weight of n-gram score.
        fuzzy_topk_logits: Reduce the logits tensor to match the tensor that the solution
            is based on.
    """

    def __init__(
        self,
        model: RNNT,
        blank_idx: int,
        tokenizer: Tokenizer,
        beam_width: int = 4,
        max_inputs_per_batch: int = int(1e7),
        max_symbols_per_step: Optional[int] = 8,
        max_symbol_per_sample: Optional[int] = None,
        temperature: float = 1.4,
        pred_net_cache_size: int = 100,
        beam_prune_score_thresh: Union[int, float] = 0.4,
        beam_prune_topk_thresh: Union[int, float] = 1.5,
        ngram_info: Optional[NgramInfo] = None,
        fuzzy_topk_logits: bool = False,
    ):
        super().__init__(
            model=model,
            blank_idx=blank_idx,
            max_inputs_per_batch=max_inputs_per_batch,
            max_symbol_per_sample=max_symbol_per_sample,
            max_symbols_per_step=max_symbols_per_step,
            temperature=temperature,
        )
        assert beam_width > 0
        self.beam_width = beam_width
        self.score = lambda x: x.score / x.y_length_tot

        self.detokenize = tokenizer.sentpiece.id_to_piece
        self.tokenizer = tokenizer
        self.spu_unicode = SPU_UNICODE
        self.pred_net_cache_size = pred_net_cache_size

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

    def _inner_decode(
        self, f: torch.Tensor, f_len: torch.Tensor
    ) -> Tuple[List[int], List[int], None]:
        """
        Run decoding loop given encoder features.

        f is shape (time, 1, enc_dim)
        """

        training_state = self.model.training
        self.model.eval()

        device = f.device

        blank_tensor = torch.tensor([self.blank_idx], device=device, dtype=torch.long)

        sos_hyp = init_sos_hyp(self._SOS, self.ngram_lm)
        kept_hyps = {sos_hyp.hashval: sos_hyp}

        pred_net_cache = FIFOCache(self.pred_net_cache_size)

        responses: dict[int, FrameResponses] = {}
        for time_idx in range(f_len):
            # Stop if the length of the best hypothesis exceeds max_symbol_per_sample
            if self.max_symbol_per_sample is not None:
                max_hyp = max(kept_hyps.values(), key=lambda x: x.score)
                if max_hyp.y_length_tot > self.max_symbol_per_sample:
                    break

            # f_t is the acoustic encoding for this timestep, shape (1, 1, enc_dim)
            f_t = f[time_idx, :, :].unsqueeze(0)

            kept_hyps = self._beam_run_timestep(
                f_t, kept_hyps, time_idx, pred_net_cache, device, blank_tensor
            )
            responses[time_idx], kept_hyps = self.serialiser.frame_responses(
                kept_hyps, time_idx
            )

            for hyp in kept_hyps.values():
                hyp.check(self.tokenizer)

        self.model.train(training_state)

        responses[time_idx + 1] = self.serialiser.last_frame_response(kept_hyps)

        # For now, generate the timesteps and y_sequence from the final responses
        # In future, we will return the responses directly
        finals = [x.final for x in responses.values() if x.final]

        y_seqs = [final.alternatives[0].y_seq for final in finals]
        y_sequence = [x for xs in y_seqs for x in xs]

        timesteps_nest = [final.alternatives[0].timesteps for final in finals]
        timesteps = [x for xs in timesteps_nest for x in xs]

        for i, times in enumerate(timesteps_nest[:-1]):
            assert times[-1] <= timesteps_nest[i + 1][0], (
                f"Sanity check: finals' timesteps should be monotonic but there is an "
                f"overlap for {i}th and {i + 1}th: "
                f"{timesteps_nest[i]=}, {timesteps_nest[i + 1]}"
            )
        return y_sequence, timesteps, None

    def _beam_run_timestep(
        self,
        f_t: torch.Tensor,
        hyps: Dict[int, Hypothesis],
        time_idx: int,
        pred_net_cache: FIFOCache,
        device: torch.device,
        blank_tensor: torch.Tensor,
    ) -> Dict[int, Hypothesis]:
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

            g, pred_state_prime, _ = self._pred_step_cached(
                max_hyp, device, pred_net_cache
            )

            logp = self._joint_step(
                f_t, g, log_normalize=True, fuzzy=self.fuzzy_topk_logits
            )[0, :]

            # Determine whether we will expand this hypothesis with non-blank tokens
            add_ys = not self.max_symbols or (max_hyp.y_len_t < self.max_symbols)

            if add_ys and max_hyp.y_length_tot > 1:
                logp, lm_state_prime = self._lm_correction(
                    logp, f_t, g, max_hyp, device
                )
            else:
                lm_state_prime = None

            steps = self._prepare_steps(logp, blank_tensor, add_ys)

            for klogp, kidx in zip(*steps):
                hyps, kept_hyps = self._update_hyps(
                    klogp,
                    kidx,
                    max_hyp,
                    kept_hyps,
                    hyps,
                    time_idx,
                    pred_state_prime,
                    lm_state_prime,
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

    def _pred_step_cached(
        self, max_hyp: Hypothesis, device: torch.device, pred_net_cache: FIFOCache
    ) -> Tuple[
        Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]], None
    ]:
        """
        Run prediction network on hypothesis, or retrieve from cache if previously computed.
        """
        key = max_hyp.hashval
        if key in pred_net_cache:
            return pred_net_cache[key]
        result = self._pred_step(max_hyp.y_last, max_hyp.pred_state, device)
        pred_net_cache[key] = result
        return result

    def _prepare_steps(
        self, logp: torch.Tensor, blank_tensor: torch.Tensor, add_ys: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare hypothesis expansion steps.
        """
        if add_ys:
            top_k = logp.topk(self.beam_width)

            max_score = top_k[0].max()

            idxs_to_keep = top_k[0] >= max_score - self.beam_prune_topk_thresh

            # Prepare steps as modified top_k tuple and its logprob. Each entry
            # in steps represents one possible next step in RNN-T decoding.
            steps = (top_k[0][idxs_to_keep], top_k[1][idxs_to_keep])

            # Add blank if not present in the top_k. This is rare for a trained
            # model at large beam sizes, but can occur with a small beam size.
            if self.blank_idx not in steps[1]:
                steps = (
                    torch.cat((steps[0], logp[self.blank_idx].unsqueeze(0))),
                    torch.cat((steps[1], blank_tensor)),
                )

            steps = (steps[0].cpu(), steps[1].cpu())
        else:
            # Only prepare the blank
            steps = (
                logp[self.blank_idx].unsqueeze(0).cpu(),
                blank_tensor.cpu(),
            )
        return steps

    def _update_hyps(
        self,
        klogp: torch.Tensor,
        kidx: torch.Tensor,
        max_hyp: Hypothesis,
        kept_hyps: Dict[int, Hypothesis],
        hyps: Dict[int, Hypothesis],
        time_idx: int,
        pred_state_prime: Optional[Tuple[torch.Tensor, torch.Tensor]],
        lm_state_prime: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[Dict[int, Hypothesis], Dict[int, Hypothesis]]:
        """
        Create a new hypothesis and update the beam

        Hypothesis merging is done by checking the hash of the new hypothesis against those
        in the kept_hyps/hyps dicts.
        """
        new_hyp = max_hyp.clone()
        new_hyp.score += float(klogp)

        if kidx.item() == self.blank_idx:
            if new_hyp.hashval in kept_hyps:
                kept_hyps[new_hyp.hashval].score = np.logaddexp(
                    kept_hyps[new_hyp.hashval].score, new_hyp.score
                )
            else:
                kept_hyps[new_hyp.hashval] = new_hyp
        else:
            new_hyp.timesteps.append(time_idx)
            new_hyp.pred_state = pred_state_prime
            new_hyp.lm_state = lm_state_prime
            new_hyp.y_seq.append(int(kidx))
            new_hyp.y_len_t += 1

            new_hyp = self._ngram_correction(max_hyp.ngram_lm_state, new_hyp)

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
        max_hyp_score = self.score(max(hyps.values(), key=self.score))
        return {
            k: v
            for k, v in hyps.items()
            if self.score(v) >= max_hyp_score - self.beam_prune_score_thresh
        }

    def _sort_nbest(self, hyps: ValuesView[Hypothesis]) -> List[Hypothesis]:
        """Sort hypotheses by length normalized score."""
        return sorted(hyps, key=self.score, reverse=True)

    def _lm_correction(
        self,
        logp: torch.Tensor,
        f_t: torch.Tensor,
        g: Optional[torch.Tensor],
        max_hyp: Hypothesis,
        device: torch.device,
    ) -> Tuple[torch.Tensor, None]:
        """Is a no-op because language models are not supported."""
        return logp, None
