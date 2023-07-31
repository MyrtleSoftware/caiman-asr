# Copyright (c) 2022 Myrtle.ai
# written by iria [& rob] @ myrtle, Jul 2022

import math
from typing import Final

import torch
from torch import nn

from rnnt_train.common.hard_activation_functions import Hardsigmoid, Hardtanh
from rnnt_train.common.quantize import BfpQuantizer, BrainFloatQuantizer, NullClass


class CustomLSTM(nn.Module):
    """
    CustomLSTM is a partial drop-in replacement for the standard PyTorch LSTM
    which implements all of its internal calculations using simpler PyTorch
    classes and functions rather than calling _VF.  Weight names match those in
    the standard PyTorch LSTM enabling transfer of state_dict and checkpoints.

    CustomLSTM supports multiple layers, dropout, quantization, and the use of hard
    activation functions.  It always uses bias weights and sequence-first input
    tensors. It requires batched input tensors (seq_len, batch_size, input_size).
    It does not support bidirectional LSTMs or projection layers.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        dropout=0.0,
        hard=False,
        quantize=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hard = hard
        self.quantize = quantize
        if dropout != 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        # create and register the weights & biases of the LSTM layer by layer
        for layer in range(num_layers):
            if layer == 0:
                layer_input_size = input_size
            else:
                layer_input_size = hidden_size
            # create
            U = nn.Parameter(torch.Tensor(4 * hidden_size, layer_input_size))
            V = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
            bih = nn.Parameter(torch.Tensor(4 * hidden_size))
            bhh = nn.Parameter(torch.Tensor(4 * hidden_size))
            # register
            self.register_parameter(name=f"weight_ih_l{layer}", param=U)
            self.register_parameter(name=f"weight_hh_l{layer}", param=V)
            self.register_parameter(name=f"bias_ih_l{layer}", param=bih)
            self.register_parameter(name=f"bias_hh_l{layer}", param=bhh)
        # initialize the weights & biases
        rsh = 1.0 / math.sqrt(hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-rsh, rsh)
        # Most of the compute takes place inside the CustomLSTMLayer class.
        # The CustomLSTMLayer class is therefore wrapped in TorchScript for speed.
        # When using quantization during inference, TorchScript is not used.
        # This is because TorchScript only supports a limited number of types
        # (as seen here: https://pytorch.org/docs/stable/jit_language_reference.html)
        # and doesn't support custom classes, as are defined in the QPyTorch package.
        if self.quantize:
            self.custom_layer = CustomLSTMLayer(hidden_size, hard, quantize)
        else:
            self.custom_layer = torch.jit.script(
                CustomLSTMLayer(hidden_size, hard, quantize)
            )

    def forward(self, input_tensor, init_states=None):
        # input_tensor is (seq_len, batch, input_size)
        seq_len, batch_size, input_size = input_tensor.size()
        # init_states is None (defaults to zeros) or (h_0, c_0) where
        # h_0 is (num_layers, batch, hidden_size) initial hidden state
        # c_0 is (num_layers, batch, hidden_size) initial cell   state
        # accumulate layer final states in lists
        h_fl = []
        c_fl = []
        # apply the LSTM layer by layer
        for layer in range(self.num_layers):
            # determine layer input
            if layer == 0:
                layer_input = input_tensor  # (seq_len, batch, input_size)
            else:
                if self.dropout:
                    layer_input = self.dropout(output)  # (seq_len, batch, hidden_size)
                else:
                    layer_input = output  # (seq_len, batch, hidden_size)
            # determine layer h_0, c_0
            if init_states == None:
                h_0 = torch.zeros(batch_size, self.hidden_size).to(input_tensor.device)
                c_0 = torch.zeros(batch_size, self.hidden_size).to(input_tensor.device)
            else:
                h_0 = init_states[0][layer]  # (batch, hidden_size)
                c_0 = init_states[1][layer]  # (batch, hidden_size)
            # determine layer weights
            U = getattr(
                self, f"weight_ih_l{layer}"
            )  # (4*hidden_size, layer_input_size)
            V = getattr(self, f"weight_hh_l{layer}")  # (4*hidden_size, hidden_size)
            bih = getattr(self, f"bias_ih_l{layer}")  # (4*hidden_size, )
            bhh = getattr(self, f"bias_hh_l{layer}")  # (4*hidden_size, )
            # apply torchscript layer
            output, (h_f, c_f) = self.custom_layer(
                layer_input, h_0, c_0, U, V, bih, bhh
            )
            # accumulate the layer final states
            h_fl.append(h_f)
            c_fl.append(c_f)

        # compute the final state tensors
        h_f = torch.stack(h_fl, dim=0)  # (num_layers, batch, hidden_size)
        c_f = torch.stack(c_fl, dim=0)  # (num_layers, batch, hidden_size)
        # return the final layer output sequence         (seq_len, batch, hidden_size)
        # and the final hidden and cell state tensors    (num_layers, batch, hidden_size)
        return output, (h_f, c_f)


# Using Torchscript (jit) speeds up execution, see https://pytorch.org/docs/stable/jit.html
# Using Torchscript on the whole Module, lstm = torch.jit.script(CustomLSTM(arguments)) was
# not possible under PyTorch 1.7.0 due to a jit/deepcopy/Parameter bug which had not yet
# been resolved, see https://github.com/pytorch/pytorch/issues/44951.  However, using
# Torchscript on just a single Parameter-free LSTM layer is possible.


class CustomLSTMLayer(nn.Module):
    # Final declarations help Torchscript optimize the module
    HS: Final[int]
    hard: Final[bool]
    quantize: Final[bool]

    def __init__(self, hidden_size, hard, quantize):
        super().__init__()
        self.HS = hidden_size
        self.hard = hard
        self.quantize = quantize
        # setup the activation functions
        if self.hard:
            self.tanh = Hardtanh
            self.sigmoid = Hardsigmoid
        else:
            self.tanh = torch.tanh
            self.sigmoid = torch.sigmoid
        # setup the quantization functions
        if self.quantize:
            self.bf16 = BrainFloatQuantizer(
                fp_exp=8, fp_man=7, forward_rounding="nearest"
            )
            self.bfp_dim0 = BfpQuantizer(
                dim=0, block_size=8, fp_exp=8, fp_man=7, forward_rounding="nearest"
            )
            self.bfp_dim1 = BfpQuantizer(
                dim=1, block_size=8, fp_exp=8, fp_man=7, forward_rounding="nearest"
            )
        else:
            self.bf16 = NullClass()
            self.bfp_dim0 = NullClass()
            self.bfp_dim1 = NullClass()

    def forward(self, layer_input, h_t, c_t, U, V, bih, bhh):
        # layer_input is (seq_len, batch, layer_input_size)
        # h_t is (batch, HS) initial hidden state
        # c_t is (batch, HS) initial cell   state
        # U, V, bih, bhh shapes as above
        seq_len, batch_size, layer_input_size = layer_input.size()
        #
        output_seq = []
        for t in range(seq_len):
            x_t = layer_input[t]  # (batch, layer_input_size)
            # @ is matrix multiply
            # The gates are calculated by the formula below
            # gates = x_t @ U.t() + h_t @ V.t() + bih + bhh # (batch, 4*HS)
            # But this is split into parts, so that the appropriate quantization
            # functions are applied if necessary.
            Ut = U.t()  # transpose of weight tensor U
            Vt = V.t()  # transpose of weight tensor V
            x_t = self.bfp_dim1(x_t)  # BF16 & BFP of input tensor
            h_t = self.bfp_dim1(h_t)  # BF16 & BFP of hidden state tensor
            Ut = self.bfp_dim0(Ut)  # BF16 & BFP of weight tensor Ut
            Vt = self.bfp_dim0(Vt)  # BF16 & BFP of weight tensor Vt
            bih = self.bf16(bih)  # BF16       of bias   tensor bih
            bhh = self.bf16(bhh)  # BF16       of bias   tensor bhh
            # construct the gates tensor
            xU = x_t @ Ut
            hV = h_t @ Vt
            gates = xU + hV + bih + bhh
            gates = self.bf16(gates)

            i_t, f_t, g_t, o_t = gates.chunk(4, 1)  # all (batch, HS)
            i_t = self.sigmoid(i_t)  # (batch, HS) input gate
            f_t = self.sigmoid(f_t)  # (batch, HS) forget gate
            g_t = self.tanh(g_t)  # (batch, HS) input candidates
            o_t = self.sigmoid(o_t)  # (batch, HS) output gate

            # quantize gates after application of sigmoid/tanh functions
            i_t = self.bf16(i_t)  # (batch, HS) input  gate
            f_t = self.bf16(f_t)  # (batch, HS) forget gate
            g_t = self.bf16(g_t)  # (batch, HS) input candidates
            o_t = self.bf16(o_t)  # (batch, HS) output gate

            # calculate cell and hidden states and apply quantization
            c_t = f_t * c_t + i_t * g_t  # (batch, HS) cell state
            c_t = self.bf16(c_t)
            h_t = o_t * self.tanh(c_t)  # (batch, HS) hidden state (and current output)
            h_t = self.bf16(h_t)
            output_seq.append(h_t)
        #
        output = torch.stack(output_seq, dim=0)  # (seq_len, batch, HS)
        return output, (h_t, c_t)  # (seq_len, batch, HS), ((batch, HS), (batch, HS))
