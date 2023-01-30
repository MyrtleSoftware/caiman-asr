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
    def __init__(self, blank_idx, packed_input):
        super().__init__()
        self.t_loss = TransducerLoss(packed_input=packed_input)
        self.blank_idx = blank_idx

    def forward(self, logits, logit_lens, y, y_lens, batch_offset, max_f_len):
        """
        Computes the RNNT loss function
        
        Args:
            inputs:
                logits: logits tensor of size [B, T, U, K+1] or [batch, max_seq_len,
                max_output_seq, vocab_size+1]. Precision is float32, or float16 when 
                AMP=true.
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

        loss = self.t_loss(logits,
                            y,
                            logit_lens,
                            y_lens,
                            self.blank_idx,
                            batch_offset=batch_offset,
                            max_f_len=max_f_len,
                            ).mean()
        return loss

