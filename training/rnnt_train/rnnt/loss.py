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

import torch
from apex.contrib.transducer import TransducerLoss


class apexTransducerLoss(torch.nn.Module):
    """
    NVIDIA apex RNNT implementation from the apex module transducer_loss_cuda.
    """

    def __init__(self, blank_idx, packed_input, validate_first_n_remaining=10):
        super().__init__()
        self.t_loss = TransducerLoss(packed_input=packed_input)
        self.packed_input = packed_input
        self.blank_idx = blank_idx
        self.validate_first_n_remaining = validate_first_n_remaining

    def forward(self, logits, logit_lens, y, y_lens, batch_offset, max_f_len):
        """
        Computes the RNNT loss function

        Args:
            inputs:
                logits: logits tensor of size [B, T, U, K+1] or [batch, max_seq_len,
                max_output_seq, vocab_size+1] if packed_input=False. Otherwise, logits
                is a tensor of size [total_packed, K+1] where total_packed = batch_offset[-1].
                Precision is float32, or float16 when AMP=true.
                logit_lens: the length of logits tensor.
                y: text transcription tensor.
                y_lens: the length text tensor.
                batch_offset: cumulative sum of T*(U+1).
                max_f_len: max number of features.
        """

        if y.dtype != torch.int32:
            y = y.int()

        if logit_lens.dtype != torch.int32:
            logit_lens = logit_lens.int()

        if y_lens.dtype != torch.int32:
            y_lens = y_lens.int()

        if self.validate_first_n_remaining > 0:
            # Inputs validation is expensive as it requires sending items from GPU -> CPU
            # and waiting for the result to arrive. Therefore we only validate the first
            # ten times this loss is called. We don't just validate once because there
            # was a bug that was only caught by validating multiple times.
            self._validate_inputs(logits, batch_offset, y)
            self.validate_first_n_remaining -= 1

        loss = self.t_loss(
            logits,
            y,
            logit_lens,
            y_lens,
            self.blank_idx,
            batch_offset=batch_offset,
            max_f_len=max_f_len,
        ).mean()
        return loss

    def _validate_inputs(self, logits, batch_offset, y) -> None:
        """
        Validate consistency of inputs.
        """
        if self.packed_input:
            assert len(logits.shape) == 2, (
                "When packed_input=True, logits should be of shape [total_packed, K+1] "
                f"but {logits.shape=}"
            )
            total_packed, _ = logits.shape
            assert total_packed == batch_offset[-1].item(), (
                "Packed input shape and batch_offsets are inconsistent. Should have "
                "total_packed == batch_offset[-1] but "
                f"{total_packed} != {batch_offset[-1].item()}"
            )
        else:
            assert len(logits.shape) == 4, (
                "When packed_input=False, logits should be of shape [B, T, U, K+1] "
                f"but {logits.shape=}"
            )
            assert (
                y.shape[1] == logits.shape[2] - 1
            ), f"When packed_input=False, {y.shape[1]=} should be 1 less than {logits.shape[2]=}"


def get_packing_meta_data(
    feat_lens: torch.Tensor,
    txt_lens: torch.Tensor,
    enc_time_reduction: int,
) -> dict:
    """
    Return packing metadata for TransducerLoss and TransducerJoint
    """
    dict_meta_data = {"batch_offset": None, "max_f_len": None, "packed_batch": None}
    final_feat_lens = (feat_lens + enc_time_reduction - 1) // enc_time_reduction
    batch_offset = torch.cumsum(final_feat_lens * (txt_lens + 1), dim=0)
    max_f_len = max(final_feat_lens)

    dict_meta_data = {
        "batch_offset": batch_offset,
        "max_f_len": max_f_len,
        "packed_batch": batch_offset[-1].item(),
    }

    return dict_meta_data
