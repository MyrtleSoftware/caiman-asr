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

from rnnt_train.rnnt.model import label_collate
from rnnt_train.rnnt.sub_models import RNNTSubModels


class RNNTDecoder(ABC):
    def __init__(
        self,
        blank_idx,
        max_symbol_per_sample,
        max_symbols_per_step,
        lm_info,
        temperature=1.0,
    ):
        self.blank_idx = blank_idx
        assert max_symbols_per_step is None or max_symbols_per_step > 0
        self.max_symbols = max_symbols_per_step
        assert max_symbol_per_sample is None or max_symbol_per_sample > 0
        self.max_symbol_per_sample = max_symbol_per_sample
        self._SOS = -1  # start of sequence token
        self.lm_info = lm_info
        self.temperature = temperature

    def _pred_step(self, model, label, hidden, device):
        if label == self._SOS:
            return model.predict(None, hidden, add_sos=False)

        label = label_collate([[label]]).to(device)
        return model.predict(label, hidden, add_sos=False)

    def _joint_step(self, model, enc, pred, log_normalize=False):
        logits = model.joint(enc, pred)[:, 0, 0, :]

        if log_normalize:
            logprobs = F.log_softmax(
                logits / self.temperature, dim=len(logits.shape) - 1
            )
            return logprobs
        else:
            return logits

    @beartype
    def decode(
        self, model, feats, feat_lens
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Returns a list of tokenized sentences and list of timestamps given an input batch.

        If timestamps are not returned output_timestamps is empty.

        Args:
            model     : RNN-T model to use for decoding
            feats     : a tensor of logmels, shape (seq_len, batch, feat_dim)
            feat_lens : list of int representing the length of each sequence of
                features, shape (batch,)

        Returns:
            list of lists of decoded tokens, one sequence for each member of the batch
            list of lists of per-token timestamps or an empty list
        """
        model = model.rnnt if isinstance(model, RNNTSubModels) else model
        model = getattr(model, "module", model)
        with torch.no_grad():
            # encs     is shape (batch, time, enc_dim)
            # enc_lens is shape (batch,)
            encs, enc_lens, _ = model.encode(feats, feat_lens)

            output = []
            output_timestamps = []
            for batch_idx in range(encs.size(0)):
                # this_enc is shape (time, 1, enc_dim)
                this_enc = encs[batch_idx, :, :].unsqueeze(1)
                this_len = enc_lens[batch_idx]
                sentence, timestamps = self._inner_decode(model, this_enc, this_len)
                output.append(sentence)
                if timestamps is not None:
                    output_timestamps.append(timestamps)
        return output, output_timestamps

    @abstractmethod
    def _inner_decode(self, model, x, x_len) -> tuple:
        pass
