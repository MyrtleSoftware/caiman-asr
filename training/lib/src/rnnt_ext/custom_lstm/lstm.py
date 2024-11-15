# Note: Any modifications to this code require recompilation for the changes to take effect.
import math

import rnnt_ext.cuda.lstm as lstm_cu
import torch
from beartype import beartype
from beartype.typing import Optional, Tuple
from torch import Tensor as Ten


class Function(torch.autograd.Function):
    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.cuda.amp.custom_fwd
    @beartype
    def forward(
        ctx,
        lstm_fused_fwd,
        lstm_fused_bwd,
        y0: Ten,
        c0: Ten,
        x: Ten,
        W: Ten,
        R: Ten,
        bW: Ten,
        bR: Ten,
    ) -> Tuple[Ten, Ten]:
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
            Tuple of (y_all, c_all) where output is a stack
            of the hidden states y1...yn. y_all and c_all contain all the hidden
            and cell states. Gradients are calculated for output only.
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
        ctx.lstm_fused_bwd = lstm_fused_bwd

        # y[-1] and c[-1] are the final hidden/cell states, y[1:] is a stack of
        # hidden states 1...n.

        return y[1:], c[1:]

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.cuda.amp.custom_bwd
    @beartype
    def backward(
        ctx,
        delta: Ten,
        *_,
    ) -> Tuple[None, None, None, None, Optional[Ten], Ten, Ten, Ten, Ten]:
        """
        Compute the backwards pass of an LSTM layer.

        Never call this function directly, instead call .backward(..) on the output
        of the .apply(..) method.

        Arguments:
            delta: Gradient of the loss with respect to the output of the layer.
            _: Ignored argument(s) to match the return signature of the
                forward function.

        Returns:
            A gradient for each of the inputs to the forward function.
        """

        W, Rp, x, y, c, gates = ctx.saved_variables
        lstm_fused_bwd = ctx.lstm_fused_bwd

        assert delta.dtype == Rp.dtype

        dG: Ten = torch.empty_like(gates, memory_format=torch.contiguous_format)

        lstm_fused_bwd(Rp, gates, c, delta, dG)

        dB: Ten = dG.sum([0, 1])

        # Must do manual bmm to get performance
        dG: Ten = dG.flatten(0, 1)

        shape_dX = (delta.shape[0], x.shape[0] // delta.shape[0], x.shape[1])

        dX: Optional[Ten] = (
            torch.matmul(dG, W).view(shape_dX) if x.requires_grad else None
        )

        dW: Ten = torch.matmul(dG.t(), x.detach())
        # NOTE: In https://doi.org/10.1109/TNNLS.2016.2582924 they sum dG from 1 to
        # N and y from 0 to N-1 but their y is shifted by 1.
        dR: Ten = torch.matmul(dG.t(), y)

        return None, None, None, None, dX, dW, dR, dB.unsqueeze(0), dB.unsqueeze(0)


class HardLayer(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return Function.apply(
            lstm_cu.lstm_fused_fwd_hard, lstm_cu.lstm_fused_bwd_hard, *args, **kwargs
        )


class SoftLayer(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return Function.apply(
            lstm_cu.lstm_fused_fwd_soft, lstm_cu.lstm_fused_bwd_soft, *args, **kwargs
        )


class Layer(torch.nn.Module):
    @beartype
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hard: bool = False,
        rw_dropout: float = 0.0,
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
        self.layer_fun = HardLayer() if hard else SoftLayer()
        self.rw_dropout = rw_dropout
        self.drop_fun = (
            torch.nn.Dropout(p=rw_dropout) if rw_dropout != 0.0 else torch.nn.Identity()
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

    @beartype
    def forward(self, x: Ten, state: Tuple[Ten, Ten]) -> Tuple[Ten, Ten]:
        """
        Compute the forward pass of an LSTM layer.

        Arguments:
            x: Input sequence.
            state: Tuple of (y0, c0) where y0 is the initial hidden state and c0 is the
                initial cell state.

        Returns:
            Tuple of (y, c) where: y is a stack of the hidden states y1...yn
            and c is a stack of the cell states c1...cn.
        """

        # Tensor float is enabled by default for cudnn, so mirror that here.
        cache_tf32 = torch.backends.cuda.matmul.allow_tf32

        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32

        h, c = self.layer_fun(
            *state,
            x,
            self.weight_ih,
            self.drop_fun(self.weight_hh),
            self.bias_ih,
            self.bias_hh,
        )

        torch.backends.cuda.matmul.allow_tf32 = cache_tf32

        return h, c

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

    @beartype
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
            torch.nn.Dropout(p=self.bl_dropout)
            if self.bl_dropout != 0.0
            else torch.nn.Identity()
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

    @beartype
    def forward(
        self, input: Ten, state: Optional[Tuple[Ten, Ten]] = None
    ) -> Tuple[Ten, Tuple[Ten, Ten], Tuple[Ten, Ten]]:
        """
        Compute the forward pass of a multilayer LSTM.

        Do not call this function directly, instead use the function call operator on
        the CustomLSTM object.

        Arguments:
            input: Input sequence.
            state: Tuple of (h_0, c_0) where h_0 is the initial hidden state and
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

        # Element i of this list will be a tensor of all hidden states from layer i
        all_h_fl = []
        # Same, but for cell states
        all_c_fl = []

        x = None

        for i, layer in enumerate(self.layers):
            # Determine layer input.
            layer_input = input if i == 0 else self.drop_function(x)

            # Determine layer h_0, c_0
            if state is None:
                shape = [e for e in input.shape[1:]]
                shape[-1] = self.hidden_size

                h_0 = torch.zeros(shape, device=input.device, dtype=input.dtype)
                c_0 = torch.zeros(shape, device=input.device, dtype=input.dtype)
            else:
                h_0 = state[0][i].detach()
                c_0 = state[1][i].detach()

            # Apply custom layer
            h, c = layer(layer_input, (h_0, c_0))

            # Accumulate the layer final states
            h_fl.append(h[-1])
            c_fl.append(c[-1])

            # Accumulate all states from the layer
            all_h_fl.append(h)
            all_c_fl.append(c)

            # Final hidden is the output of each layer
            x = h

        h_f = torch.stack(h_fl, dim=0)
        c_f = torch.stack(c_fl, dim=0)

        all_h_f = torch.stack(all_h_fl, dim=0)
        all_c_f = torch.stack(all_c_fl, dim=0)

        return x, (h_f, c_f), (all_h_f, all_c_f)
