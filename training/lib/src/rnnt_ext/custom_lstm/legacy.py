# written by iria [& rob] @ myrtle, Jul 2022

import math

import torch
from beartype.typing import Final
from einops import rearrange
from torch import nn


# The 8 in Hardsigmoid is easier/cheaper in hardware than non-powers-of-2
def Hardsigmoid(x):
    return torch.clamp(0.5 + x / 8.0, min=0.0, max=1.0)


# Using Hardtanh in a @torch.jit.script was faster than using torch.nn.functional.hardtanh
def Hardtanh(x):
    return torch.clamp(x, min=-1.0, max=1.0)


class CustomLSTM(nn.Module):
    """
    CustomLSTM is a partial drop-in replacement for the standard PyTorch LSTM
    which implements all of its internal calculations using simpler PyTorch
    classes and functions rather than calling _VF.  Weight names match those in
    the standard PyTorch LSTM enabling transfer of state_dict and checkpoints.

    CustomLSTM supports multiple layers, between-layer dropout, quantization,
    the use of hard activation functions, and recurrent-weight dropout.  It
    always uses bias weights and sequence-first input tensors. It requires
    batched input tensors (seq_len, batch_size, input_size).  It does not
    support bidirectional LSTMs or projection layers.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        dropout=0.0,
        hard=False,
        quantize=False,
        rw_dropout=0.0,
    ):
        super().__init__()
        self.input_size = input_size  # parameter name matches PyTorch LSTM
        self.hidden_size = hidden_size  # parameter name matches PyTorch LSTM
        self.num_layers = num_layers  # parameter name matches PyTorch LSTM
        self.bl_dropout = (
            dropout  # (between layer) dropout parameter name must match PyTorch LSTM
        )
        self.hard = hard  # CustomLSTM specific : use hard activation functions
        self.quantize = quantize  # CustomLSTM specific : quantize values
        self.rw_dropout = rw_dropout  # CustomLSTM specific : recurrent weight dropout
        # setup between layer dropout function
        if self.bl_dropout != 0.0:
            self.bld_func = nn.Dropout(p=self.bl_dropout)
        else:
            self.bld_func = None
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
        # set up the custom layer
        if self.quantize:
            # When using quantization TorchScript is not used.  This is because
            # TorchScript only supports a limited number of types, which does not
            # include the custom classes defined in the QPyTorch package, see
            # https://pytorch.org/docs/stable/jit_language_reference.html
            self.custom_layer = CustomLSTMLayer(hard, quantize, rw_dropout)
        else:
            # Most of the compute takes place inside the CustomLSTMLayer class.
            # This class is therefore wrapped in TorchScript (jit) for speed.
            # Under PyTorch 1.13 Automatic Mixed Precision (AMP) is not compatible
            # with TorchScript.  We follow https://pytorch.org/docs/1.13/amp.html
            # quote "For now, we suggest to disable the Jit Autocast Pass".
            # Setting autocast to False (below) is 34% faster than the True
            # default when using the non-quantizing CustomLSTM with AMP on A100s.
            # However, on A100s it is 5% faster still to simply not use AMP.
            # Note that the first 3 iterations when using jit are very slow.
            torch._C._jit_set_autocast_mode(False)
            self.custom_layer = torch.jit.script(
                CustomLSTMLayer(hard, quantize, rw_dropout)
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
        if self.training:
            # Element i of this list will be a tensor of all hidden states from layer i
            all_h_fl = []
            # Same, but for cell states
            all_c_fl = []

        # apply the LSTM layer by layer
        for layer in range(self.num_layers):
            # determine layer input
            if layer == 0:
                layer_input = input_tensor  # (seq_len, batch, input_size)
            else:
                if self.bld_func:
                    layer_input = self.bld_func(output)  # (seq_len, batch, hidden_size)
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
            # apply custom layer
            # Get the output, final states, and all the states
            output, (h_f, c_f), (all_h_f, all_c_f) = self.custom_layer(
                layer_input, h_0, c_0, U, V, bih, bhh
            )
            # accumulate the layer final states
            h_fl.append(h_f)
            c_fl.append(c_f)

            if self.training:
                # Accumulate all states from the layer
                all_h_fl.append(all_h_f)
                all_c_fl.append(all_c_f)

        # compute the final state tensors
        h_f = torch.stack(h_fl, dim=0)  # (num_layers, batch, hidden_size)
        c_f = torch.stack(c_fl, dim=0)  # (num_layers, batch, hidden_size)
        # return the final layer output sequence          (seq_len, batch, hidden_size)
        # and the final hidden and cell state tensors     (num_layers, batch, hidden_size)

        if self.training:
            # Turn list of tensors into just tensors
            all_h_f = rearrange(
                all_h_fl,
                "num_layers seq_len batch hidden_size -> num_layers seq_len batch hidden_size",
            )
            all_c_f = rearrange(
                all_c_fl,
                "num_layers seq_len batch hidden_size -> num_layers seq_len batch hidden_size",
            )
            all_hidden = (all_h_f, all_c_f)
        else:
            all_hidden = None

        return output, (h_f, c_f), all_hidden


# Using an identity function instead of a class simplifies the code from print(custom_layer.code)
def identity_func(x):
    return x


# Using Torchscript (jit) speeds up execution, see https://pytorch.org/docs/stable/jit.html
# Using Torchscript on the whole Module, lstm = torch.jit.script(CustomLSTM(arguments)) is
# not compatible with the use of deepcopy in train.py (etc), causes problems with non-static
# variable types in the CustomLSTM code above and is not likely to bring any speedup anyway
# since all the compute is in the CustomLSTMLayer below.

# I've tried many versions of CustomLSTMLayer, including those which pass model weights to
# the init() and those which pass layer_input_size and hidden_size (marked Final) to the
# init() and define and register all weight Parameters locally.  These bring no speedup,
# and also conflict with the use of deepcopy elsewhere.  I've also tried PyTorch 1.13
# FuncTorch which also doesn't help.  For now, this is probably as good as it gets - rob.


class CustomLSTMLayer(nn.Module):
    # Final declarations help Torchscript optimize the module
    hard: Final[bool]
    quantize: Final[bool]
    rw_dropout: Final[float]

    def __init__(self, hard, quantize, rw_dropout):
        super().__init__()
        self.hard = hard
        self.quantize = quantize
        self.rw_dropout = rw_dropout
        # setup the activation functions
        if self.hard:
            self.tanh = Hardtanh
            self.sigmoid = Hardsigmoid
        else:
            self.tanh = torch.tanh
            self.sigmoid = torch.sigmoid
        # setup the quantization functions
        if self.quantize:
            # this is a slow import so we do it only when necessary
            from rnnt_train.common.quantize import BfpQuantizer, BrainFloatQuantizer

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
            self.bf16 = identity_func
            self.bfp_dim0 = identity_func
            self.bfp_dim1 = identity_func
        # setup the recurrent weight dropout function
        if self.rw_dropout:
            self.rwd_func = nn.Dropout(p=self.rw_dropout)
        else:
            self.rwd_func = identity_func

    def forward(self, layer_input, h_t, c_t, U, V, bih, bhh):
        # layer_input is (seq_len, batch, layer_input_size)
        # hidden_size is HS for brevity
        # h_t is (batch, HS) initial hidden state
        # c_t is (batch, HS) initial cell   state
        # U, V, bih, bhh shapes as above
        seq_len, batch_size, layer_input_size = layer_input.size()

        # prepare weight tensors
        V = self.rwd_func(V)  # recurrent weight dropout
        Ut = U.t()  # transpose of weight tensor U
        Vt = V.t()  # transpose of weight tensor V
        Ut = self.bfp_dim0(Ut)  # BF16 & BFP of weight tensor Ut
        Vt = self.bfp_dim0(Vt)  # BF16 & BFP of weight tensor Vt
        bih = self.bf16(bih)  # BF16       of bias   tensor bih
        bhh = self.bf16(bhh)  # BF16       of bias   tensor bhh

        # Element i of this list is the hidden state at time i in the sequence
        all_h_tl = []
        # Same, but for cell states
        all_c_tl = []

        output_seq = []
        for t in range(seq_len):
            x_t = layer_input[t]  # (batch, layer_input_size)
            x_t = self.bfp_dim1(x_t)  # BF16 & BFP of input tensor
            h_t = self.bfp_dim1(h_t)  # BF16 & BFP of hidden state tensor

            # @ is matrix multiply
            # The gates are calculated using the formula
            # gates = x_t @ U.t() + h_t @ V.t() + bih + bhh
            # But this is split into parts to reflect any quantization used
            xU = x_t @ Ut  # BFP @ BFP
            hV = h_t @ Vt  # BFP @ BFP
            gates = xU + hV + bih + bhh  # (batch, 4*HS)
            gates = self.bf16(gates)  # (batch, 4*HS)

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
            all_h_tl.append(h_t)
            all_c_tl.append(c_t)
        #
        output = torch.stack(output_seq, dim=0)  # (seq_len, batch, HS)

        # Turn list of tensors into just tensors
        # Can't use einops.rearrange because incompatible with torch.jit
        all_h_t = torch.stack(all_h_tl, dim=0)
        all_c_t = torch.stack(all_c_tl, dim=0)

        # The shape of the return values is:
        # (seq_len, batch, HS), ((batch, HS), (batch, HS), (seq_len, batch, HS), (seq_len, batch, HS))
        return (
            output,
            (h_t, c_t),
            (all_h_t, all_c_t),
        )
