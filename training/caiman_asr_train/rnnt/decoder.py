# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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


from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import Dict, List, Optional

from caiman_asr_train.rnnt.eos_strategy import (
    EOSBlank,
    EOSIgnore,
    EOSPredict,
    EOSStrategy,
)
from caiman_asr_train.rnnt.fuzzy_logits import get_topk_logits
from caiman_asr_train.rnnt.model import label_collate
from caiman_asr_train.rnnt.response import FrameResponses
from caiman_asr_train.rnnt.sub_models import RNNTSubModels
from caiman_asr_train.rnnt.unbatch_encoder import encode_lower_batch_size
from caiman_asr_train.train_utils.distributed import unwrap_ddp


class RNNTDecoder(ABC):
    @beartype
    def __init__(
        self,
        model,
        eos_strategy: EOSStrategy,
        blank_idx: int,
        max_inputs_per_batch: int = int(1e7),
    ):
        model = model.rnnt if isinstance(model, RNNTSubModels) else model
        model = unwrap_ddp(model)

        self.model = model
        self.max_inputs_per_batch = max_inputs_per_batch
        self.eos_strategy = eos_strategy
        self.blank_idx = blank_idx

    @torch.no_grad()
    @beartype
    def decode(
        self,
        feats: torch.Tensor,
        feat_lens: torch.Tensor,
    ) -> List[Dict[int, FrameResponses]]:
        """
        Returns a list of frame responses dictionaries for each frame in the batch.

        Args:
            feats     : a tensor of logmels, shape (seq_len, batch, feat_dim)
            feat_lens : list of int representing the length of each sequence of
                features, shape (batch,)

        Returns:
            list of dict[int, FrameResponses], where the outer list is over
            the batch dimension and the dictionary keys are the frame indices.
        """

        # encs     is shape (batch, time, enc_dim)
        # enc_lens is shape (batch,)
        encs, enc_lens = encode_lower_batch_size(
            self.model, feats, feat_lens, self.max_inputs_per_batch
        )

        return self._inner_decode(encs, enc_lens)

    @property
    @beartype
    def eos_index(self) -> Optional[int]:
        """
        Test if the decoder will predict the EOS token.

        Returns: The index of the EOS token if it is predicted, None otherwise.
        """
        match self.eos_strategy:
            case EOSPredict(idx, _, _):
                return idx
            case _:
                return None

    @abstractmethod
    @beartype
    def _inner_decode(
        self, encs: torch.Tensor, enc_lens: torch.Tensor
    ) -> List[Dict[int, FrameResponses]]:
        pass


class RNNTCommonDecoder(RNNTDecoder):
    @beartype
    def __init__(
        self,
        model,
        blank_idx: int,
        eos_strategy: EOSStrategy,
        max_symbol_per_sample: Optional[int],
        max_symbols_per_step: Optional[int],
        max_inputs_per_batch: int = int(1e7),
        temperature: float = 1.0,
        # This isn't user-configurable for the greedy decoder.
        # This is configurable for the beam decoder, and the default is 1.4 for beam
    ):
        super().__init__(model, eos_strategy, blank_idx, max_inputs_per_batch)

        assert max_symbols_per_step is None or max_symbols_per_step > 0
        assert max_symbol_per_sample is None or max_symbol_per_sample > 0

        self.blank_idx = blank_idx
        self.max_symbols = max_symbols_per_step
        self.max_symbol_per_sample = max_symbol_per_sample
        self._SOS = -1  # start of sequence token
        self.temperature = temperature

    def _pred_step_raw(self, label, hidden):
        return self.model.predict(label, hidden, add_sos=False)

    def _pred_step(self, label, hidden, device=None):
        if label == self._SOS:
            return self.model.predict(None, hidden, add_sos=False)

        label = label_collate([[label]]).to(device)
        return self._pred_step_raw(label, hidden)

    def _eos_prob_correction(self, logprobs):
        match self.eos_strategy:
            case None:
                pass
            case EOSIgnore(idx):
                logprobs[:, idx] = -float("inf")
            case EOSBlank(idx):
                logprobs[:, self.blank_idx] = torch.logaddexp(
                    logprobs[:, self.blank_idx], logprobs[:, idx]
                )
                logprobs[:, idx] = -float("inf")
            case EOSPredict(idx, alpha, beta):
                logprobs[:, idx] = logprobs[:, idx] * alpha
                if beta > 0:
                    logprobs[:, idx] = torch.where(
                        logprobs[:, idx] > np.log(beta), logprobs[:, idx], -float("inf")
                    )

        return logprobs

    def _joint_step(self, enc, pred, fuzzy=False):
        logits = self.model.joint(enc, pred)[:, 0, 0, :]

        if fuzzy:
            logits = get_topk_logits(logits)

        # The normalization must have happened before the EOS
        # beta strategy is applied for beta to have a meaningful
        # value.
        logprobs = F.log_softmax(logits / self.temperature, dim=-1)

        return self._eos_prob_correction(logprobs)

    @abstractmethod
    @beartype
    def _inner_decode(
        self, encs: torch.Tensor, enc_lens: torch.Tensor
    ) -> List[Dict[int, FrameResponses]]:
        pass
