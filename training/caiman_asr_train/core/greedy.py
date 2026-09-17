"""
A minimal, single-utterance greedy transducer decoder.

This mirrors the decoding semantics of
``caiman_asr_train.rnnt.batched_greedy.RNNTBatchedGreedyDecoder`` (including
encoder/prediction-network prefix handling and the ``max_symbols_per_step`` /
``max_symbol_per_sample`` stop conditions) without any of the batching
machinery, and without the optional n-gram, keyword-boosting and EOS
strategies.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import List, Optional, Tuple
from torch import Tensor

from caiman_asr_train.core.batches import TimeFirst
from caiman_asr_train.core.model import RNNT, StatePred


@beartype
@dataclass
class GreedyResult:
    """
    The decode of a single utterance.

    Members:
        tokens:     The non-blank token ids, in emission order.
        timestamps: The encoder frame index each token was emitted at.
        probs:      The model's probability for each token.
    """

    tokens: List[int]
    timestamps: List[int]
    probs: List[float]


class GreedyDecoder:
    @beartype
    def __init__(
        self,
        model: RNNT,
        blank_idx: Optional[int] = None,
        max_symbols_per_step: Optional[int] = 30,
        max_symbol_per_sample: Optional[int] = None,
        prefix_token_ids: Optional[List[int]] = None,
    ):
        """
        A single-utterance greedy transducer decoder.

        Args:
            model: The transducer model to use for decoding. Its LSTM backend
                must return threaded states from ``predict`` (hardware
                checkpoints loaded via ``core.load_hw`` do).
            blank_idx: Index of the blank symbol; defaults to the end of the
                vocab (``n_classes - 1``).
            max_symbols_per_step: Force a frame advance after this many
                consecutive non-blank emissions; None means no limit. With no
                limit a model that never emits blank will never terminate.
            max_symbol_per_sample: The maximum number of (non-blank) symbols
                that can be decoded per utterance; None means no limit.
            prefix_token_ids: Token IDs to use as prefix during inference. The
                prefix always primes the prediction network; it is also fed to
                the encoder iff the model has a ``prefix_token_map``.
        """

        assert max_symbols_per_step is None or max_symbols_per_step > 0
        assert max_symbol_per_sample is None or max_symbol_per_sample > 0

        self.model = model
        self.blank_idx = model.config.n_classes - 1 if blank_idx is None else blank_idx
        self.max_symbols_per_step = max_symbols_per_step
        self.max_symbol_per_sample = max_symbol_per_sample
        self.prefix_token_ids = prefix_token_ids

    @torch.no_grad()
    @beartype
    def decode(self, feats: Tensor) -> GreedyResult:
        """
        Decode a single utterance of features.

        Args:
            feats: A tensor of shape (time, in_feats). The time dimension must
                be a (non-zero) multiple of the model's enc_stack_time_factor.
        """

        assert feats.ndim == 2, f"Expected (time, in_feats) got shape {feats.shape}"

        n = self.model.config.enc_stack_time_factor
        assert feats.shape[0] > 0 and feats.shape[0] % n == 0, (
            f"Feature length {feats.shape[0]} must be a non-zero multiple of "
            f"the enc_stack_time_factor ({n})"
        )
        assert feats.shape[1] == self.model.config.in_feats, (
            f"Expected {self.model.config.in_feats} features per frame, "
            f"got {feats.shape[1]}"
        )

        was_training = self.model.training
        self.model.eval()

        try:
            feats = feats.to(self._device)

            if self.model.config.prefix_token_map is not None:
                # If model has token_map assume it requires an encoder prefix.
                enc_prefix = self._prefix_time_first()
            else:
                enc_prefix = None

            feats_tf = TimeFirst(
                data=feats.unsqueeze(1),
                lens=torch.tensor([feats.shape[0]], device=feats.device),
            )

            encs, _ = self.model.encode(state=None, feats=feats_tf, prefix=enc_prefix)

            return self._greedy_loop(encs.data[:, 0])
        finally:
            self.model.train(was_training)

    @torch.no_grad()
    @beartype
    def decode_encodings(self, encs: Tensor) -> GreedyResult:
        """
        Decode a single utterance from pre-computed encodings.

        Args:
            encs: A tensor of shape (enc_time, joint_n_hid): the encoder
                output after the ``joint_enc`` projection, i.e. exactly what
                ``model.encode(...)`` returns.
        """

        assert (
            encs.ndim == 2 and encs.shape[0] > 0
        ), f"Expected (enc_time, joint_n_hid) got shape {encs.shape}"
        assert encs.shape[1] == self.model.config.joint_n_hid, (
            f"Expected {self.model.config.joint_n_hid} joint features, "
            f"got {encs.shape[1]}"
        )

        was_training = self.model.training
        self.model.eval()

        try:
            return self._greedy_loop(encs.to(self._device))
        finally:
            self.model.train(was_training)

    @property
    def _device(self) -> torch.device:
        return self.model.joint_enc.weight.device

    @beartype
    def _prefix_time_first(self) -> Optional[TimeFirst]:
        if not self.prefix_token_ids:
            return None

        # Lens must be int32: predict() accumulates them into an int32 tensor.
        return TimeFirst(
            data=torch.tensor(self.prefix_token_ids, device=self._device).unsqueeze(1),
            lens=torch.tensor(
                [len(self.prefix_token_ids)], dtype=torch.int32, device=self._device
            ),
        )

    @beartype
    def _greedy_loop(self, encs: Tensor) -> GreedyResult:
        """
        Run the greedy decoding loop over encodings of shape (T, joint_n_hid).
        """

        T = encs.shape[0]

        # Prime the prediction network with the prefix if available.
        g, state = self.model.predict(
            state=self.model.sos_for(1),
            prefix=self._prefix_time_first(),
            tokens=None,
        )

        assert state is not None, (
            "The prediction network did not return a threaded state; the greedy "
            "decoder needs an LSTM backend that does, e.g. the "
            "TorchConfig(return_all=True) backend used for hardware checkpoints."
        )

        # The predictor's output for the last consumed token (SoS or prefix).
        g_vec = g.data[-1, 0]

        t = 0
        # Non-blank emissions since the last forced advance. As in the batched
        # decoder this is deliberately NOT reset on a blank-advance, so it can
        # accumulate across frames until it saturates.
        step_symbols = 0

        tokens: List[int] = []
        timestamps: List[int] = []
        probs: List[float] = []

        while True:
            logprobs = F.log_softmax(self.model.joint_net(encs[t] + g_vec), dim=-1)
            k = int(logprobs.argmax())

            last_frame = t == T - 1

            # Stop conditions, mirroring _update_done in the batched decoder.
            if last_frame and k == self.blank_idx:
                break
            if (
                (max_step := self.max_symbols_per_step) is not None
                and last_frame
                and step_symbols >= max_step
            ):
                break
            if (max := self.max_symbol_per_sample) is not None and len(tokens) >= max:
                break

            if k != self.blank_idx:
                tokens.append(k)
                timestamps.append(t)
                probs.append(logprobs[k].exp().item())

            # Advance if we predicted a blank.
            advance = k == self.blank_idx

            # Advance if we hit the max symbols per step.
            if max_step is not None:
                step_symbols += k != self.blank_idx
                if step_symbols >= max_step:
                    advance = True
                    # Keep the saturated count at the last frame (where the
                    # advance is clamped) so the stop condition above fires on
                    # the next iteration; reset it otherwise.
                    if not last_frame:
                        step_symbols = 0

            if advance and not last_frame:
                t += 1

            # Run the prediction network on the new non-blank.
            if k != self.blank_idx:
                g_vec, state = self._predict_step(k, state)

        return GreedyResult(tokens=tokens, timestamps=timestamps, probs=probs)

    @beartype
    def _predict_step(self, token: int, state: StatePred) -> Tuple[Tensor, StatePred]:
        """
        Advance the prediction network by one (non-blank) token, returning its
        new output vector (joint_n_hid,) and threaded state.
        """

        tokens = TimeFirst(
            data=torch.tensor([[token]], device=self._device),
            lens=torch.ones(1, dtype=torch.int32, device=self._device),
        )

        out, new_state = self.model.predict(state=state, prefix=None, tokens=tokens)

        assert new_state is not None

        # Ignore the SoS output, i.e. the +1 in U + 1.
        return out.data[-1, 0], new_state
