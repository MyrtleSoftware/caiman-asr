#!/usr/bin/env python3
# Copyright (c) 2023, Myrtle Software Limited, www.myrtle.ai. All rights reserved.


from itertools import count

import torch
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple
from torch import Tensor

from caiman_asr_train.data.tokenizer import Tokenizer
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
        """
        super().__init__(
            model=model,
            blank_idx=blank_idx,
            eos_strategy=eos_strategy,
            max_inputs_per_batch=max_inputs_per_batch,
            max_symbol_per_sample=max_symbol_per_sample,
            max_symbols_per_step=max_symbols_per_step,
        )

        self.detokenize = tokenizer.sentpiece.id_to_piece

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

        # Make initial prediction/hidden state.
        g, (h, c), _ = self._pred_step(self._SOS, None)

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

        labels = []
        timestamps = []
        label_probs = []

        max_offset = enc_lens.view(-1, 1, 1) - 1

        while True:
            # Build this steps view of the encodings.
            torch.gather(encs, 1, enc_offset.expand(-1, -1, jH), out=f)
            # Make a prediction for every batch
            logprobs = self._joint_step(f, g, fuzzy=False)
            logprobs, k = logprobs.max(-1, keepdim=True)

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
