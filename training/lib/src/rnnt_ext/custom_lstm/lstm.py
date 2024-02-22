# Note: Any modifications to this code require recompilation for the changes to take effect.
import math

import lstm_cu
import torch
from beartype.typing import Optional, Tuple
from einops import rearrange
from torch import Tensor as Ten


def make_layer_function(lstm_fused_fwd, lstm_fused_bwd):
    """
    Build a custom LSTM layer function using the provided fused kernels.
    """

    class Function(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd
        def forward(
            ctx, y0: Ten, c0: Ten, x: Ten, W: Ten, R: Ten, bW: Ten, bR: Ten
        ) -> Tuple[Ten, Tuple[Ten, Ten], Tuple[Ten, Ten]]:
            """
            Compute the forward pass of an LSTM layer.

            Never call this function directly, instead use the .apply(..) method which
            will provide the context object.

            Arguments:
                ctx: Context object for backpropagation.
                y0: Initial hidden state.
                c0: Initial cell state.
                x: Input sequence.
                W: Input weight matrix.
                R: Recurrent weight matrix.
                bW: Input bias vector.
                bR: Recurrent bias vector.

            Returns:
                Tuple of (output, (yn, cn), (y_all, c_all))) where output is a stack of
                    the hidden states y1...yn.
                y_all and c_all contain all the hidden and cell states
            """

            # Stacks sequences along the batch dimension (and make contiguous).
            batch_view_x: Ten = x.flatten(0, 1)
            # Now compute all the xW^t's + biases in one go.
            gates: Ten = torch.addmm(bW + bR, batch_view_x, W.t())
            # Then unfold the sequences back into their own dimensions.
            gates: Ten = gates.view(x.shape[0], x.shape[1], W.shape[0])
            # Above equiv to torch.baddbmm(bW + bR, x, W.t().expand(x.shape[0], -1, -1))

            # This reports if it is required to compute dX in backwards.
            batch_view_x.requires_grad = x.requires_grad

            # If in autocasting region, the gates could be float16.
            # Hence, propagate gates's dtype from now on.

            shape = [i for i in x.shape]
            shape[-1] = W.shape[0] // 4
            shape[0] += 1

            kwargs = {
                "dtype": gates.dtype,
                "device": x.device,
                "memory_format": torch.contiguous_format,
            }

            # Store all the intermediate hidden/cell states in one tensor.

            y: Ten = torch.empty(shape, **kwargs)
            c: Ten = torch.empty(shape, **kwargs)

            y[0].copy_(y0)
            c[0].copy_(c0)

            # Lift autocast's casting outside the C++ loop
            Rp = R.type(dtype=gates.dtype)

            lstm_fused_fwd(Rp, gates, c, y)

            ctx.save_for_backward(W, Rp, batch_view_x, y[:-1].flatten(0, 1), c, gates)

            # y[-1] and c[-1] are the final hidden/cell states, y[1:] is a stack of
            # hidden states 1...n.

            return y[1:], (y[-1], c[-1]), (y[1:], c[1:])

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(
            ctx,
            delta: Ten,
            _ignore: None,
            _ignore2: None,
        ) -> Tuple[None, None, Optional[Ten], Ten, Ten, Ten, Ten]:
            """
            Compute the backwards pass of an LSTM layer.

            Never call this function directly, instead call .backward(..) on the output
            of the .apply(..) method.

            Arguments:
                delta: Gradient of the loss with respect to the output of the layer.
                _ignore: Ignored argument to match the return signature of the forward
                    function.
                _ignore2: As above

            Returns:
                A gradient for each of the inputs to the forward function.
            """

            W, Rp, x, y, c, gates = ctx.saved_variables

            assert _ignore is None and delta.dtype == Rp.dtype
            assert _ignore2 is None

            dG: Ten = torch.empty_like(gates, memory_format=torch.contiguous_format)

            lstm_fused_bwd(Rp, gates, c, delta, dG)

            dB: Ten = dG.sum([0, 1])

            # Must do manual bmm to get performance
            dG: Ten = dG.flatten(0, 1)

            shape_dX = (delta.shape[0], x.shape[0] // delta.shape[0], x.shape[1])

            dX: Optional[Ten] = (
                torch.matmul(dG, W).view(shape_dX) if x.requires_grad else None
            )

            # Just in case it tries to keep track of some gradients.
            x.requires_grad = False

            dW: Ten = torch.matmul(dG.t(), x)
            # NOTE: In https://doi.org/10.1109/TNNLS.2016.2582924 they sum dG from 1 to
            # N and y from 0 to N-1 but their y is shifted by 1.
            dR: Ten = torch.matmul(dG.t(), y)

            return None, None, dX, dW, dR, dB.unsqueeze(0), dB.unsqueeze(0)

    return Function


hard_layer_fun = make_layer_function(
    lstm_cu.lstm_fused_fwd_hard, lstm_cu.lstm_fused_bwd_hard
)

soft_layer_fun = make_layer_function(
    lstm_cu.lstm_fused_fwd_soft, lstm_cu.lstm_fused_bwd_soft
)


class Layer(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        hard=False,
        rw_dropout=0.0,
        dtype=None,
        device=None,
    ):
        """Layer is a single LSTM layer with custom activation functions.

        Arguments:
            input_size: The number of features in the input x.
            hidden_size: The number of features in the hidden state h.
            hard: If True, use hard activation functions else use soft.
            rw_dropout: Recurrent weight dropout probability.
            dtype: The data type of the weights and biases.
            device: The device to place the weights and biases on.
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hard = hard
        self.layer_fun = hard_layer_fun if hard else soft_layer_fun
        self.rw_dropout = rw_dropout
        self.drop_fun = (
            torch.nn.Dropout(p=rw_dropout) if rw_dropout != 0.0 else lambda x: x
        )

        kwargs = {
            "dtype": dtype,
            "device": device,
            "requires_grad": True,
            "memory_format": torch.contiguous_format,
        }

        self.weight_ih = torch.nn.Parameter(
            torch.empty(4 * hidden_size, input_size, **kwargs)
        )

        self.weight_hh = torch.nn.Parameter(
            torch.empty(4 * hidden_size, hidden_size, **kwargs)
        )

        self.bias_ih = torch.nn.Parameter(torch.empty(4 * hidden_size, **kwargs))
        self.bias_hh = torch.nn.Parameter(torch.empty(4 * hidden_size, **kwargs))

        rsh = 1.0 / math.sqrt(hidden_size)

        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-rsh, rsh)

    def forward(self, x: Ten, state: Tuple[Ten, Ten]) -> Tuple[Ten, Tuple[Ten, Ten]]:
        """
        Compute the forward pass of an LSTM layer.

        Arguments:
            x: Input sequence.
            state: Tuple of (y0, c0) where y0 is the initial hidden state and c0 is the
                initial cell state.

        Returns:
            Tuple of (output, (yn, cn), (y_all, c_all))) where output is a stack of the
                hidden states y1...yn.
            y_all and c_all contain all the hidden and cell states
        """

        # Tensor float is enabled by default for cudnn, so mirror that here.
        cache_tf32 = torch.backends.cuda.matmul.allow_tf32

        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32

        output, (hn, cn), (all_hn, all_cn) = self.layer_fun.apply(
            *state,
            x,
            self.weight_ih,
            self.drop_fun(self.weight_hh),
            self.bias_ih,
            self.bias_hh,
        )

        torch.backends.cuda.matmul.allow_tf32 = cache_tf32

        return output, (hn, cn), (all_hn, all_cn)

    def extra_repr(self):
        return (
            f"input_size={self.input_size:.>4}, hidden_size={self.hidden_size:.>4}, "
            f"hard={self.hard}, rw_dropout={self.rw_dropout}"
        )


class CustomLSTM(torch.nn.Module):
    """
    CustomLSTM is a partial drop-in replacement for the standard PyTorch LSTM
    which supports the use of hard activation functions.  Weight names match those
    in the standard PyTorch LSTM enabling transfer of state_dict and checkpoints.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        hard: bool = False,
        quantize: bool = False,
        rw_dropout: float = 0.0,
        dtype=None,
        device=None,
    ):
        """CustomLSTM is a partial drop-in replacement for the standard PyTorch LSTM

        Arguments:
            input_size: The number of expected features in the input x.
            hidden_size: The number of features in the hidden state h.
            num_layers: Number of recurrent layers.
            dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM
                layer except the last layer, with dropout probability equal to dropout.
            hard: If True, use hard activation functions else use soft.
            quantize: Unsupported (must be False), here for api compatibility with
                Legacy CustomLSTM.
            rw_dropout: Recurrent weight dropout probability.
            dtype: The data type of the weights and biases.
            device: The device to place the weights and biases on.

        """
        super().__init__()

        assert not quantize, "Cuda CustomLSTM does not support quantization"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bl_dropout = dropout

        self.quantize = quantize  # CustomLSTM specific : quantize weights.
        self.hard = hard  # CustomLSTM specific : use hard activation functions.
        self.rw_dropout = rw_dropout  # CustomLSTM specific : recurrent weight dropout.

        self.drop_function = (
            torch.nn.Dropout(p=self.bl_dropout) if self.bl_dropout != 0.0 else None
        )

        kwargs = {
            "hidden_size": hidden_size,
            "hard": hard,
            "rw_dropout": rw_dropout,
            "dtype": dtype,
            "device": device,
        }

        self.layers = [Layer(input_size, **kwargs)]
        self.layers.extend(Layer(hidden_size, **kwargs) for _ in range(num_layers - 1))

        # Register the weights & biases of the LSTM layer by layer
        for i, layer in enumerate(self.layers):
            self.register_parameter(name=f"weight_ih_l{i}", param=layer.weight_ih)
            self.register_parameter(name=f"weight_hh_l{i}", param=layer.weight_hh)
            self.register_parameter(name=f"bias_ih_l{i}", param=layer.bias_ih)
            self.register_parameter(name=f"bias_hh_l{i}", param=layer.bias_hh)

    def forward(self, input_tensor, init_states=None):
        """
        Compute the forward pass of a multilayer LSTM.

        Do not call this function directly, instead use the function call operator on
        the CustomLSTM object.

        Arguments:
            input_tensor: Input sequence.
            init_states: Tuple of (h_0, c_0) where h_0 is the initial hidden state and
                c_0 is the initial cell state for each layers.
                If set to None, then h_0 and c_0 default to zero.

        Returns:
            Tuple of (output, (h_n, c_n), all_hidden). output is a stack
                of the final layer's hidden states y1...yn. h_n and c_n are the
                final hidden and cell states for each layer. During validation,
                all_hidden is None. During training, all_hidden is a tuple
                (h_all, c_all), where h_all and c_all contain all the hidden and
                cell states for each layer.
        """

        h_fl = []
        c_fl = []
        if self.training:
            # Element i of this list will be a tensor of all hidden states from layer i
            all_h_fl = []
            # Same, but for cell states
            all_c_fl = []

        output = None
        for i, layer in enumerate(self.layers):
            # Determine layer input.
            if i == 0:
                layer_input = input_tensor
            else:
                if self.drop_function is None:
                    layer_input = output
                else:
                    layer_input = self.drop_function(output)

            # Determine layer h_0, c_0
            if init_states is None:
                shape = [e for e in input_tensor.shape[1:]]
                shape[-1] = self.hidden_size

                h_0 = torch.zeros(
                    shape, device=input_tensor.device, dtype=input_tensor.dtype
                )
                c_0 = torch.zeros(
                    shape, device=input_tensor.device, dtype=input_tensor.dtype
                )
            else:
                h_0 = init_states[0][i]
                c_0 = init_states[1][i]

            # Apply custom layer
            # Get the output, final states, and all the states
            output, (h_f, c_f), (all_h_f, all_c_f) = layer(layer_input, (h_0, c_0))

            # Accumulate the layer final states
            h_fl.append(h_f)
            c_fl.append(c_f)

            if self.training:
                # Accumulate all states from the layer
                all_h_fl.append(all_h_f)
                all_c_fl.append(all_c_f)

        h_f = torch.stack(h_fl, dim=0)
        c_f = torch.stack(c_fl, dim=0)

        if self.training:
            # Turn list of tensors into just tensors
            all_h_f = rearrange(
                all_h_fl,
                "num_layers seq_len batch hidden_size -> num_layers seq_len batch hidden_size",  # noqa: E501
            )
            all_c_f = rearrange(
                all_c_fl,
                "num_layers seq_len batch hidden_size -> num_layers seq_len batch hidden_size",  # noqa: E501
            )
            all_hidden = (all_h_f, all_c_f)
        else:
            all_hidden = None

        return output, (h_f, c_f), all_hidden
