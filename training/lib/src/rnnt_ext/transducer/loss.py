# Note: Any modifications to this code require recompilation for the changes to take effect.


# Copyright (c) 2024, Myrtle Software Limited. All rights reserved.
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

# This file is modified from:
#     https://github.com/NVIDIA/apex/blob/master/apex/contrib/transducer/transducer.py


import math

import rnnt_ext.cuda.logsumexp as logsumexp_cu
import rnnt_ext.cuda.transducer_loss as transducer_loss_cu
import torch
from beartype import beartype
from beartype.typing import List, Optional


class TransducerLoss(torch.nn.Module):
    """Transducer loss
    Detail of this loss function can be found in: Sequence Transduction
    with Recurrent Neural Networks.

    Arguments:
        packed_input (bool, optional): whether to pack the output in a compact
            form with don't-care data being removed. (default: False)
    """

    def __init__(
        self,
        packed_input: bool = False,
    ):
        super(TransducerLoss, self).__init__()
        self.packed_input = packed_input
        self.dummy_batch_offset = torch.empty(0)

    @beartype
    def forward(
        self,
        x: torch.Tensor,
        label: torch.Tensor,
        f_len: torch.Tensor,
        y_len: torch.Tensor,
        blank_idx: int,
        eos_idx: Optional[int] = None,
        star_idx: Optional[int] = None,
        batch_offset: Optional[torch.Tensor] = None,
        max_f_len: Optional[int] = None,
        debug_list: Optional[List[torch.Tensor]] = None,
        delay_penalty: float = 0.0,
        eos_penalty: float = 0.0,
        star_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Forward operation of transducer joint

        Arguments:
            x (tensor): input tensor to the loss function with a shape of (B, T, U, H) if
                unpacked, or (*, H) if packed.
            label (tensor): labels for the input data.
            f_len (tensor): lengths of the inputs in the time dimension for each batch.
            y_len (tensor): lengths of the labels for each batch.
            blank_idx (int): index for the null symbol.
            eos_idx (int, optional): index for the end-of-sentence symbol. (default: None)
            star_idx (int, optional): index for the uncertain symbol. (default: None)
            batch_offset (tensor, optional): tensor containing the offset of each batch
                in the input. For example, batch offset can be obtained from:
                batch_offset = torch.cumsum(f_len*(y_len+1), dim=0)
                This argument is required if packed_input == True, and is ignored if
                packed_input == False. (default: None)
            max_f_len (int, optional): maximum length of the input in the time dimension.
                For example, it can be obtained as
                max_f_len = max(f_len)
                This argument is required if packed_input == True, and is ignored if
                packed_input == False. (default: None)
                (default: None)
            debug_list (list, optional): when an empty list is supplied, Alpha and Beta
                generated in the forward operation will be attached to this list for
                debug purpose. (default: None)
            delay_penalty: a scalar to penalize delayed emission of non-blank tokens.
                The idea is to boost probabilities of non-blanks left of the diagonal
                in the RNN-T lattice. (default: 0.0)
            eos_penalty: a scalar to bias the model to emit the end-of-sentence
                token as soon as possible. (default: 0.0)
            star_penalty: a scalar to penalize repeated emission in the star channel
                this is interpreted as a probability. (default: 1.0)

        Returns:
            loss (tensor): the transducer loss with shape (B).
        """

        assert len(x.shape) == 4 or len(x.shape) == 2, "Shape (B, T, U, H) or (*, H)"

        assert f_len.min() >= 1, "f_len must be non-negative"
        assert y_len.min() >= 0, "y_len must be non-negative"
        assert y_len.max() <= label.size(1), "y_len must be less than label length"

        if self.packed_input:
            if batch_offset is None or max_f_len is None:
                raise Exception(
                    "Please specify batch_offset and max_f_len when packing is enabled"
                )
            my_batch_offset = batch_offset
            my_max_f_len = max_f_len

            assert my_max_f_len == f_len.max()
        else:
            my_batch_offset = self.dummy_batch_offset
            my_max_f_len = x.size(1)

        return TransducerLossFunc.apply(
            x,
            label,
            f_len,
            y_len,
            my_batch_offset,
            delay_penalty,
            my_max_f_len,
            blank_idx,
            eos_penalty,
            eos_idx,
            math.log(star_penalty),
            star_idx,
            debug_list,
            self.packed_input,
        )


class TransducerLossFunc(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(
        ctx,
        x,
        label,
        f_len,
        y_len,
        batch_offset,
        delay_penalty,
        max_f_len,
        blank_idx,
        eos_penalty,
        eos_idx,
        star_penalty,
        star_idx,
        debug_list,
        packed_input,
    ):
        if packed_input:
            denom = logsumexp_cu.logsumexp(x, 128, True)
        else:
            assert x.is_contiguous(), "activations must be contiguous or packed"
            # View ND as 2D, compute logsumexp along the last dimension, then
            # view back to (N - 1)D.
            denom = logsumexp_cu.logsumexp(x.view(-1, x.shape[-1]), 128, True).view(
                x.shape[:-1]
            )

        assert denom.shape == x.shape[:-1]

        if eos_idx is None:
            eos_idx = -1
        else:
            assert eos_idx != blank_idx, "eos_idx must be different from blank_idx"

        if star_idx is None:
            star_idx = -2
        else:
            assert star_idx != blank_idx, "star_idx must be different from blank_idx"

        assert star_idx != eos_idx, "star_idx must be different from eos_idx"

        alpha, beta, loss = transducer_loss_cu.forward(
            x,
            denom,
            label,
            f_len,
            y_len,
            batch_offset,
            delay_penalty,
            max_f_len,
            blank_idx,
            eos_penalty,
            eos_idx,
            star_penalty,
            star_idx,
            packed_input,
        )

        if debug_list == []:
            debug_list += [alpha, beta]
        ctx.save_for_backward(x, denom, alpha, beta, f_len, y_len, label, batch_offset)
        ctx.blank_idx = blank_idx
        ctx.eos_penalty = eos_penalty
        ctx.eos_idx = eos_idx
        ctx.star_penalty = star_penalty
        ctx.star_idx = star_idx

        ctx.packed_input = packed_input
        ctx.max_f_len = max_f_len
        ctx.delay_penalty = delay_penalty

        return loss

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, loss_grad):
        x, denom, alpha, beta, f_len, y_len, label, batch_offset = ctx.saved_tensors

        x_grad = transducer_loss_cu.backward(
            x,
            denom,
            loss_grad,
            alpha,
            beta,
            f_len,
            y_len,
            label,
            batch_offset,
            ctx.delay_penalty,
            ctx.max_f_len,
            ctx.blank_idx,
            ctx.eos_penalty,
            ctx.eos_idx,
            ctx.star_penalty,
            ctx.star_idx,
            ctx.packed_input,
        )

        return x_grad, *([None] * 13)
