#!/usr/bin/env python3
# Copyright (c) 2023, Myrtle Software Limited, www.myrtle.ai. All rights reserved.


import os
from itertools import chain, count

import kenlm
import torch
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple
from torch import Tensor

from caiman_asr_train.data.tokenizer import Tokenizer
from caiman_asr_train.keywords.process import load_keywords
from caiman_asr_train.lm.kenlm_ngram import KenLmModel, NgramInfo
from caiman_asr_train.rnnt.decoder import RNNTCommonDecoder
from caiman_asr_train.rnnt.eos_strategy import EOSStrategy
from caiman_asr_train.rnnt.response import (
    DecodingResponse,
    FrameResponses,
    HypothesisResponse,
)


class RNNTBatchedGreedyDecoder(RNNTCommonDecoder):
    @beartype
    def __init__(
        self,
        model,
        blank_idx: int,
        eos_strategy: EOSStrategy,
        max_inputs_per_batch: int,
        tokenizer: Tokenizer,
        max_symbols_per_step: Optional[int] = 30,
        max_symbol_per_sample: Optional[int] = None,
        keywords_path: Optional[str] = None,
        boost_keywords: bool = False,
        keyword_boost_c0: float = 0.3,
        keyword_boost_beta: float = 0.9,
        keyword_boost_context_score: float = 1.0,
        ngram_info: Optional[NgramInfo] = None,
        top_k: Optional[int] = 5,
        prefix_token_ids: Optional[List[int]] = None,
    ):
        """A batched greedy transducer decoder.

        Args:
            model: The transducer model to use for decoding.
            blank_idx : Which is assumed to be at the end of the vocab.
            eos_strategy: The EOS token handling strategy.
            max_symbols_per_step: The maximum number of symbols that can be added
                to a sequence in a single time step; if set to None then there is
                no limit. This includes the blank symbol.
            max_symbol_per_sample: The maximum number of (non-blank) symbols that
                can be decoded per utterance; if set to None then there is no limit
            ngram_info: NgramInfo dataclass containing file path to n-gram language model
                and weight of n-gram score.
            top_k: number of candidates k to consider for ngram updates
            prefix_token_ids: List of token IDs to use as prefix during inference
        """
        super().__init__(
            model=model,
            blank_idx=blank_idx,
            eos_strategy=eos_strategy,
            max_inputs_per_batch=max_inputs_per_batch,
            max_symbol_per_sample=max_symbol_per_sample,
            max_symbols_per_step=max_symbols_per_step,
            prefix_token_ids=prefix_token_ids,
        )

        if boost_keywords:
            self.keywords = load_keywords(
                keywords_path,
                keyword_boost_c0,
                keyword_boost_beta,
                keyword_boost_context_score,
            )
        else:
            self.keywords = None

        if ngram_info:
            ngram_path, self.ngram_alpha = ngram_info.path, ngram_info.scale_factor
            assert os.path.isfile(
                ngram_path
            ), f"N-gram LM path {ngram_path} does not exist."
            assert (
                self.ngram_alpha >= 0.0
            ), f"N-gram scale factor is negative, {self.ngram_alpha}"
            self.ngram_lm = KenLmModel(ngram_path)
            self.top_k = top_k
        else:
            self.ngram_lm = None

        self.detokenize = tokenizer.sentpiece.id_to_piece
        self._detokenized_vocab = [
            self.detokenize(i) for i in range(tokenizer.num_labels)
        ]

    @torch.no_grad()
    @beartype
    def _inner_decode(
        self, encs: Tensor, enc_lens: Tensor
    ) -> List[Dict[int, FrameResponses]]:
        """
        Run decoding loop given encoder features.

        encs: A tensor of encodings, shape (batch, time, enc_dim).

        enc_lens: A tensor representing the length of each sequence of
                  encodings, shape (batch,).
        """

        training_state = self.model.training
        self.model.eval()

        B, _, jH = encs.shape  # Here jH = joint's hidden dimension.

        # Make initial prediction/hidden state with prefix if available
        g, (h, c), _ = self._pred_step_with_prefix(device=encs.device, batch_size=B)

        g = g.expand(B, -1, -1).contiguous()  # (B, 1, jH)
        h = h.expand(-1, B, -1).contiguous()  # (L, B, pH)
        c = c.expand(-1, B, -1).contiguous()  # (L, B, pH)

        L, _, pH = h.shape  # Here pH = prediction's hidden dimension.

        enc_offset = torch.zeros((B, 1, 1), dtype=torch.int64, device=encs.device)
        any_tok_per_step = torch.zeros_like(enc_offset)
        nb_per_sample = torch.zeros_like(enc_offset)
        advance = torch.zeros_like(enc_offset)
        done = torch.zeros_like(enc_offset, dtype=torch.bool)
        f = torch.empty(B, 1, jH, dtype=encs.dtype, device=encs.device)

        if self.ngram_lm:
            ngram_lm_states = self.ngram_lm.get_batch_initial_states(B)
            ngram_states_cache = {}

        if self.keywords:
            keyword_states = [self.keywords.init()] * B
            keyword_states_cache = {}

        labels = []
        timestamps = []
        label_probs = []

        max_offset = enc_lens.view(-1, 1, 1) - 1

        while True:
            # Build this steps view of the encodings.
            torch.gather(encs, 1, enc_offset.expand(-1, -1, jH), out=f)
            # Make a prediction for every batch
            all_logprobs = self._joint_step(f, g, fuzzy=False)
            logprobs, k = all_logprobs.max(-1, keepdim=True)

            if self.ngram_lm is not None:
                all_logprobs, ngram_states_cache = self._ngram_correction(
                    all_logprobs, ngram_lm_states, ngram_states_cache, done
                )
                # make prediction based on lm biased logprobs
                lm_biased_logprobs, lm_biased_k = all_logprobs[:, :-1].max(
                    -1, keepdim=True
                )

                # update non-blanks
                torch.where(k == self.blank_idx, k, lm_biased_k, out=k)
                torch.where(
                    k == self.blank_idx, logprobs, lm_biased_logprobs, out=logprobs
                )

            if self.keywords:
                # update logprobs with keyword boosting scores
                all_logprobs, keyword_states, keyword_states_cache = (
                    self._keyword_correction(
                        all_logprobs, keyword_states, keyword_states_cache, done
                    )
                )

                # make prediction based on boosted logprobs
                boosted_logprobs, boosted_k = all_logprobs[:, :-1].max(-1, keepdim=True)

                # Update non-blanks.
                # Note: kw boosting is only applied when non-blanks are predicted.
                # A blank prediction only advances to the next frame, the prediction
                # network stays in the same position. The output/kw sequence will only
                # advance when there are non-blank predictions. Further, if boosting was
                # applied to blank predictions the model would be encouraged to omit
                # sequences, potentially causing increased deletions.
                torch.where(k == self.blank_idx, k, boosted_k, out=k)
                torch.where(
                    k == self.blank_idx, logprobs, boosted_logprobs, out=logprobs
                )

            all_done = self._update_done(
                done, enc_offset, max_offset, k, any_tok_per_step, nb_per_sample
            )

            if all_done:
                break

            # Outputs (write blank if done).
            labels.append(torch.where(done.squeeze(), self.blank_idx, k.squeeze()))
            timestamps.append(enc_offset.clone().detach().squeeze())  # Makes a copy.
            label_probs.append(logprobs.exp().squeeze())

            # Track the total non blank emissions per sample.
            if self.max_symbol_per_sample is not None:
                # Using k not labels is OK as overestimate is fine after done.
                nb_per_sample.add_(k.unsqueeze(2) != self.blank_idx)

            # Advance if we predicted a blank.
            torch.eq(k.unsqueeze(2), self.blank_idx, out=advance)

            # Advance if we hit the max symbols per step
            if (max := self.max_symbols) is not None:
                # Track the total emissions per step.
                any_tok_per_step.add_(k.unsqueeze(2) != self.blank_idx)

                # Force advance if we have hit the max emission per step count.
                torch.logical_or(advance, any_tok_per_step >= max, out=advance)
                # Reset if we've forced an advance (if we're at the end of the
                # encoder then we have not forced an advance).
                any_tok_per_step.mul_(
                    torch.logical_or(any_tok_per_step < max, enc_offset == max_offset)
                )

            enc_offset.add_(advance)
            enc_offset.clamp_(max=max_offset)

            (index_1D,) = torch.nonzero(labels[-1] != self.blank_idx, as_tuple=True)

            if len(index_1D) == 0:
                continue

            index_3D = index_1D.unsqueeze(0).unsqueeze(2).expand(L, -1, pH)

            # Extract the y's and hidden states for the non-blank indexes.
            y = torch.gather(k, 0, index_1D.unsqueeze(1))

            state = (
                torch.gather(h, 1, index_3D),
                torch.gather(c, 1, index_3D),
            )

            # Run the prediction network on the new non-blanks.
            G, (HH, CC), _ = self._pred_step_raw(y, state)

            # Update g, h, c with the prediction net's new state.
            g.scatter_(0, index_1D.unsqueeze(1).unsqueeze(2).expand(-1, -1, jH), G)
            h.scatter_(1, index_3D, HH)
            c.scatter_(1, index_3D, CC)

            if self.keywords:
                keyword_states = self._update_keyword_states(
                    index_1D, k, keyword_states
                )

            if self.ngram_lm is not None:
                ngram_lm_states = self._update_ngram_states(
                    index_1D, k, ngram_lm_states
                )

        self.model.train(training_state)

        return self._build_return_objs(enc_lens, labels, timestamps, label_probs)

    def _update_done(
        self, done, enc_offset, max_offset, k, any_tok_per_step, nb_per_sample
    ):
        """
        Update done in-place. The stop conditions are:

            (At end of encoding) && (predicting blank)
            or
            (At end of encoding) && (overflowing)
            or
            (At max non blank emissions)

        Returns True if all batches are done.
        """

        done.logical_or_(
            torch.logical_and(
                enc_offset == max_offset, k.unsqueeze(2) == self.blank_idx
            )
        )

        if (max := self.max_symbols) is not None:
            done.logical_or_(
                torch.logical_and(enc_offset == max_offset, any_tok_per_step >= max)
            )

        if (max := self.max_symbol_per_sample) is not None:
            done.logical_or_(
                nb_per_sample >= max,
            )

        return torch.all(done)

    @beartype
    def _lm_score(
        self,
        state: kenlm.State,
        ngram_list: List[str],
        ngram_states_cache: Dict[Tuple[kenlm.State, str], Tuple[float, kenlm.State]],
    ) -> Tuple[Tensor, Dict[Tuple[kenlm.State, str], Tuple[float, kenlm.State]]]:

        scores = []
        for ngram in ngram_list:
            key = (state, ngram)
            if key not in ngram_states_cache:
                score, new_state = self.ngram_lm.score_ngram(ngram, state)
                ngram_states_cache[key] = (score, new_state)
            score, _ = ngram_states_cache[key]
            scores.append(score)
        return torch.tensor(scores), ngram_states_cache

    @beartype
    def _ngram_correction(
        self,
        logprobs: Tensor,
        ngram_states: List[kenlm.State],
        ngram_states_cache: Dict[Tuple[kenlm.State, str], Tuple[float, kenlm.State]],
        done: Tensor,
    ) -> Tuple[Tensor, Dict[Tuple[kenlm.State, str], Tuple[float, kenlm.State]]]:
        """
        Apply ngram correction for active batches.
        """
        active_batches = chain.from_iterable(
            torch.nonzero(done.squeeze() == 0).tolist()
        )

        # only consider topk indices to update for each batch. For greedy decoding,
        # the impact of lm biasing beyond topk is likely to be very low
        _, topk_indices = torch.topk(logprobs[:, :-1], self.top_k, dim=1)

        for batch_id in active_batches:
            state = ngram_states[batch_id]

            token_lists = self._detokenize_batch(topk_indices[batch_id])

            lm_scores, ngram_states_cache = self._lm_score(
                state, token_lists, ngram_states_cache
            )
            lm_scores = lm_scores.to(dtype=logprobs.dtype, device=logprobs.device)
            logprobs[batch_id].index_add_(
                0, topk_indices[batch_id], self.ngram_alpha * lm_scores
            )

        return logprobs, ngram_states_cache

    @beartype
    def _update_ngram_states(
        self, index: Tensor, k: Tensor, ngram_states: List[kenlm.State]
    ) -> List[kenlm.State]:
        """
        Update ngram states for non blanks
        """
        tok_ids = [k[idx].item() for idx in index]
        toks_list = self._detokenize_batch(tok_ids)

        for i, idx in enumerate(index):
            toks = toks_list[i]
            _, ngram_states[idx] = self.ngram_lm.score_ngram(toks, ngram_states[idx])

        return ngram_states

    @beartype
    def _detokenize_batch(self, indices: torch.Tensor | List[int]) -> List[str]:
        return [self._detokenized_vocab[idx] for idx in indices]

    @beartype
    def _keyword_steps(self, token_lists: List[str], state: int) -> Tensor:
        return torch.tensor(
            [self.keywords.steps(toks, state)[0] for toks in token_lists]
        )

    @beartype
    def _keyword_correction(
        self,
        logprobs: Tensor,
        keyword_states: List[int],
        keyword_states_cache: Dict[int, Tuple[Tensor, Tensor]],
        done: Tensor,
    ) -> Tuple[Tensor, List[int], Dict[int, Tuple[Tensor, Tensor]]]:
        """
        Apply keyword correction for active batches
        """
        active_batches = chain.from_iterable(
            torch.nonzero(done.squeeze() == 0).tolist()
        )

        for batch_id in active_batches:
            state = keyword_states[batch_id]

            if state not in keyword_states_cache:

                # Get keyword scores for the vocab
                deltas = self._keyword_steps(self._detokenized_vocab, state)

                # Keep only non-zero deltas
                indices = torch.arange(len(self._detokenized_vocab))
                nonzero_mask = deltas != 0.0
                tok_idxs = indices[nonzero_mask]
                delta_vals = deltas[nonzero_mask]

                # update state cache
                keyword_states_cache[state] = (
                    tok_idxs.to(dtype=torch.int64, device=logprobs.device),
                    delta_vals.to(device=logprobs.device),
                )

            # update logprobs with boosting scores
            toks, deltas = keyword_states_cache[state]
            logprobs[batch_id].index_add_(0, toks, deltas)

        return logprobs, keyword_states, keyword_states_cache

    @beartype
    def _update_keyword_states(
        self, index: Tensor, k: Tensor, keyword_states: List[int]
    ) -> List[int]:
        """
        Update keyword states for non blanks
        """
        tok_ids = [k[idx].item() for idx in index]
        toks_list = self._detokenize_batch(tok_ids)

        for i, idx in enumerate(index):
            toks = toks_list[i]
            _, keyword_states[idx] = self.keywords.steps(toks, keyword_states[idx])

        return keyword_states

    @beartype
    def _build_return_objs(
        self,
        enc_lens: torch.Tensor,
        labels: List[torch.Tensor],
        timestamps: List[torch.Tensor],
        label_probs: List[torch.Tensor],
    ) -> List[Dict[int, FrameResponses]]:
        """
        Convert tensor outputs into response objects, we do the blank
        stripping in the tensor domain and follow the beam API by
        retuning FrameResponses(None, None) for blank frames. This also
        dramatically increases performance (compared to a blank
        DecodingResponse).
        """

        labels, timestamps, label_probs = self._transpose_and_strip(
            len(enc_lens), labels, timestamps, label_probs
        )

        out = [{} for _ in range(len(enc_lens))]

        for i, ys, ts, ps in zip(count(), labels, timestamps, label_probs):
            for y, t, p in zip(ys, ts, ps):
                if t not in out[i]:
                    out[i][t] = FrameResponses(None, final=self._hyp_response(y, t, p))
                else:
                    hyp = out[i][t].final.alternatives[0]

                    hyp.y_seq.append(y)
                    hyp.timesteps.append(t)
                    hyp.token_seq.append(self.detokenize(y))
                    hyp.confidence.append(p)

        return out

    @beartype
    def _transpose_and_strip(
        self,
        batch_size: int,
        labels: List[torch.Tensor],
        timestamps: List[torch.Tensor],
        label_probs: List[torch.Tensor],
    ) -> Tuple[List[List[int]], List[List[int]], List[List[float]]]:
        assert len(labels) == len(timestamps) == len(label_probs)

        if not labels:
            return [[]] * batch_size, [[]] * batch_size, [[]] * batch_size

        if batch_size == 1:
            # If batch size is 1 then squeeze will have removed the batch dim.
            labels = [ll.unsqueeze(0) for ll in labels]
            timestamps = [tt.unsqueeze(0) for tt in timestamps]
            label_probs = [pp.unsqueeze(0) for pp in label_probs]

        # Stack outputs into 2D tensors (joining the time axis).
        labels = torch.stack(labels, dim=1).cpu()
        timestamps = torch.stack(timestamps, dim=1).cpu()
        label_probs = torch.stack(label_probs, dim=1).cpu()

        # Unbinds along the batch dimension.
        non_blanks = [
            (s != self.blank_idx).nonzero(as_tuple=True) for s in labels.unbind()
        ]

        def strip(stack):
            return [x[ix].tolist() for x, ix in zip(stack.unbind(), non_blanks)]

        return strip(labels), strip(timestamps), strip(label_probs)

    @beartype
    def _hyp_response(self, y: int, t: int, p: float) -> DecodingResponse:
        return DecodingResponse(
            start_frame_idx=t,
            duration_frames=1,
            is_provisional=False,
            alternatives=[
                HypothesisResponse(
                    y_seq=[y],
                    timesteps=[t],
                    token_seq=[self.detokenize(y)],
                    confidence=[p],
                )
            ],
        )
