#!/usr/bin/env python3
# Copyright (c) 2023, Myrtle Software Limited, www.myrtle.ai. All rights reserved.

import numpy as np

from rnnt_train.rnnt.decoder import RNNTDecoder


class RNNTGreedyDecoder(RNNTDecoder):
    """A greedy transducer decoder.

    Args:
        blank_idx : which is assumed to be at the end of the vocab
        max_symbols_per_step: The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit
        max_symbol_per_sample: The maximum number of symbols that can be
            decoded per utterance; if set to None then there is no limit
    """

    def __init__(
        self,
        blank_idx,
        max_symbols_per_step=30,
        max_symbol_per_sample=None,
        lm_info=None,
    ):
        super().__init__(
            blank_idx=blank_idx,
            max_symbol_per_sample=max_symbol_per_sample,
            max_symbols_per_step=max_symbols_per_step,
            lm_info=lm_info,
        )

    def prepare_lm(self):
        """Is a no-op because language models not currently supported."""
        return None

    def apply_lm(
        self,
        label,
        logits,
        model,
        f,
        g,
        lm_hidden,
        device,
        k,
    ):
        """Is a no-op because language models not currently supported."""
        return k, lm_hidden

    def _inner_decode(self, model, x, x_len, dumptype, dumpidx):
        # x     is shape (time, 1, enc_dim)
        # x_len is shape ()

        device = x.device

        training_state = model.training
        model.eval()

        lm_hidden = self.prepare_lm()

        hidden = None

        label = []
        for time_idx in range(x_len):
            if (
                self.max_symbol_per_sample is not None
                and len(label) > self.max_symbol_per_sample
            ):
                break
            f = x[time_idx, :, :].unsqueeze(0)

            not_blank = True
            symbols_added = 0

            while not_blank and (
                self.max_symbols is None or symbols_added < self.max_symbols
            ):
                # run the RNNT prediction network on one step
                g, hidden_prime, _ = self._pred_step(
                    model, self._SOS if label == [] else label[-1], hidden, device
                )

                # use the RNNT joint network to compute logits, for speed
                logits = self._joint_step(model, f, g, log_normalize=False)[0, :]

                if dumptype:
                    np.save(
                        f"/results/{dumptype}logp{dumpidx}.{time_idx}.{symbols_added}.npy",
                        logits.numpy(),
                    )

                # get index k, of max logit (ie. argmax)
                v, k = logits.max(0)
                k = k.item()

                if k == self.blank_idx:
                    not_blank = False
                else:
                    # update RNN-T prediction network hidden state
                    hidden = hidden_prime

                    k, lm_hidden = self.apply_lm(
                        label, logits, model, f, g, lm_hidden, device, k
                    )

                    label.append(k)
                symbols_added += 1

        model.train(training_state)
        return label
