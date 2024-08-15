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

import torch
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import List, Tuple

from caiman_asr_train.rnnt.fuzzy_logits import get_topk_logits
from caiman_asr_train.rnnt.model import label_collate
from caiman_asr_train.rnnt.sub_models import RNNTSubModels
from caiman_asr_train.rnnt.unbatch_encoder import encode_lower_batch_size
from caiman_asr_train.train_utils.distributed import unwrap_ddp


class RNNTDecoder(ABC):
    def __init__(
        self,
        model,
        blank_idx,
        max_symbol_per_sample,
        max_symbols_per_step,
        max_inputs_per_batch: int = int(1e7),
        temperature=1.4,
        unbatch=True,
    ):
        model = model.rnnt if isinstance(model, RNNTSubModels) else model
        model = unwrap_ddp(model)
        self.model = model
        self.blank_idx = blank_idx
        assert max_symbols_per_step is None or max_symbols_per_step > 0
        self.max_symbols = max_symbols_per_step
        assert max_symbol_per_sample is None or max_symbol_per_sample > 0
        self.max_symbol_per_sample = max_symbol_per_sample
        self.max_inputs_per_batch = max_inputs_per_batch
        self._SOS = -1  # start of sequence token
        self.temperature = temperature
        self.unbatch = unbatch

    def _pred_step_raw(self, label, hidden):
        return self.model.predict(label, hidden, add_sos=False)

    def _pred_step(self, label, hidden, device=None):
        if label == self._SOS:
            return self.model.predict(None, hidden, add_sos=False)

        label = label_collate([[label]]).to(device)
        return self._pred_step_raw(label, hidden)

    def _joint_step(self, enc, pred, log_normalize=False, fuzzy=False):
        logits = self.model.joint(enc, pred)[:, 0, 0, :]

        if fuzzy:
            logits = get_topk_logits(logits)
        if log_normalize:
            logprobs = F.log_softmax(
                logits / self.temperature, dim=len(logits.shape) - 1
            )
            return logprobs
        else:
            return logits

    @beartype
    def decode(
        self,
        feats,
        feat_lens,
    ) -> Tuple[List[List[int]], List[List[int]], List[List[float]]]:
        """
        Returns a list of tokenized sentences, a list of timestamps,
        and a list of token probabilities given an input batch.

        Args:
            feats     : a tensor of logmels, shape (seq_len, batch, feat_dim)
            feat_lens : list of int representing the length of each sequence of
                features, shape (batch,)

        Returns:
            list of lists of decoded tokens, one sequence for each member of the batch
            list of lists of per-token timestamps
            list of lists of per-token probabilities or an empty list
        """

        with torch.no_grad():
            # encs     is shape (batch, time, enc_dim)
            # enc_lens is shape (batch,)
            encs, enc_lens = encode_lower_batch_size(
                self.model, feats, feat_lens, self.max_inputs_per_batch
            )

            if self.unbatch:
                output = []
                output_timestamps = []
                output_probs = []

                for batch_idx in range(encs.size(0)):
                    # this_enc is shape (1, time, enc_dim)
                    this_enc = encs[batch_idx, :, :].unsqueeze(0)
                    this_len = enc_lens[batch_idx]
                    sentence, timestamps, label_probs = self._inner_decode(
                        this_enc, this_len
                    )

                    output.append(sentence)

                    if timestamps is not None:
                        output_timestamps.append(timestamps)

                    if label_probs is not None:
                        output_probs.append(label_probs)

                return output, output_timestamps, output_probs
            else:
                return self._inner_decode(encs, enc_lens)

    @abstractmethod
    def _inner_decode(self, f, f_len) -> tuple:
        pass
