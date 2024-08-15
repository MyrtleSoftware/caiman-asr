#!/usr/bin/env python3
# Copyright (c) 2023, Myrtle Software Limited, www.myrtle.ai. All rights reserved.


import torch
from torch import Tensor

from caiman_asr_train.rnnt.decoder import RNNTDecoder


class RNNTBatchedGreedyDecoder(RNNTDecoder):
    def __init__(
        self,
        model,
        blank_idx,
        max_inputs_per_batch,
        max_symbols_per_step=30,
        max_symbol_per_sample=None,
    ):
        """A batched greedy transducer decoder.

        Args:
            model: The transducer model to use for decoding.
            blank_idx : which is assumed to be at the end of the vocab
            max_symbols_per_step: The maximum number of symbols that can be added
                to a sequence in a single time step; if set to None then there is
                no limit. This includes the blank symbol.
            max_symbol_per_sample: The maximum number of (non-blank) symbols that
                can be decoded per utterance; if set to None then there is no limit
        """
        super().__init__(
            model=model,
            blank_idx=blank_idx,
            max_inputs_per_batch=max_inputs_per_batch,
            max_symbol_per_sample=max_symbol_per_sample,
            max_symbols_per_step=max_symbols_per_step,
            unbatch=False,
        )

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
                enc_offset == max_offset, k.unsqueeze(2) >= self.blank_idx
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

    def _build_return_objs(self, batch_size, labels, timestamps, label_probs):
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
            (s < self.blank_idx).nonzero(as_tuple=True) for s in labels.unbind()
        ]

        def strip(stack):
            return [x[ix].tolist() for x, ix in zip(stack.unbind(), non_blanks)]

        return strip(labels), strip(timestamps), strip(label_probs)

    def _inner_decode(self, encs: Tensor, encs_len: Tensor):
        """
        Run decoding loop given encoder features.

        encs: A tensor of encodings, shape (batch, time, enc_dim).

        encs_len: A tensor representing the length of each sequence of
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

        max_offset = encs_len.view(-1, 1, 1) - 1

        while True:
            # Build this steps view of the encodings.
            torch.gather(encs, 1, enc_offset.expand(-1, -1, jH), out=f)
            # Make a prediction for every batch
            logits = self._joint_step(f, g, log_normalize=False)
            probs, k = logits.max(-1, keepdim=True)

            all_done = self._update_done(
                done, enc_offset, max_offset, k, any_tok_per_step, nb_per_sample
            )

            if all_done:
                break

            # Outputs (write blank if done)
            labels.append(torch.where(done.squeeze(), self.blank_idx, k.squeeze()))
            timestamps.append(enc_offset.clone().detach().squeeze())  # Makes a copy.
            probs = torch.exp(probs - torch.logsumexp(logits, dim=-1, keepdim=True))
            label_probs.append(probs.squeeze())

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

        return self._build_return_objs(B, labels, timestamps, label_probs)
