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

from caiman_asr_train.train_utils.distributed import print_once


def rnn(
    input_size,
    hidden_size,
    num_layers,
    batch_norm,
    forget_gate_bias=1.0,
    dropout=0.0,
    **kwargs,
):
    return LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_norm=batch_norm,
        dropout=dropout,
        forget_gate_bias=forget_gate_bias,
        **kwargs,
    )


class LSTM(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        batch_norm,
        dropout,
        forget_gate_bias,
        weights_init_scale=1.0,
        hidden_hidden_bias_scale=0.0,
        **kwargs,
    ):
        """Returns an LSTM with forget gate bias init to `forget_gate_bias`.

        Args:
            input_size: See `torch.nn.LSTM`.
            hidden_size: See `torch.nn.LSTM`.
            num_layers: See `torch.nn.LSTM`.
            batch_norm : Apply batch normalization after each LSTM layer.
            dropout: See `torch.nn.LSTM`.
            forget_gate_bias: For each layer and each direction, the total value of
                to initialise the forget gate bias to.

        Returns:
            An LSTM.
        """
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.batch_norm = batch_norm

        if kwargs["quantize"] and not kwargs["custom_lstm"]:
            raise ValueError(
                "Quantization `quantize: true` is not supported PyTorch's LSTM. Set "
                "`custom_lstm: true` to run quantized LSTM."
            )
        if kwargs["custom_lstm"]:
            if kwargs["quantize"]:
                print_once(
                    "WARNING : Quantization requires the (slow) legacy TorchScript "
                    "CustomLSTM\n"
                )
                # Legacy CustomLSTM import takes O(30s) on startup due to third-party
                # qtorch import
                # As such, only import when needed
                from rnnt_ext.custom_lstm.legacy import CustomLSTM
            elif kwargs["gpu_unavailable"]:
                print_once(
                    "Using the legacy TorchScript CustomLSTM because GPU unavailable\n"
                )
                if not kwargs["quantize"]:
                    # Warning message on using TorchScript CustomLSTM if no quantization
                    print_once(
                        "WARNING: Consider setting custom_lstm: false in the config file "
                        "to use the PyTorch LSTM if not applying quantization."
                    )
                from rnnt_ext.custom_lstm.legacy import CustomLSTM
            else:
                from rnnt_ext.custom_lstm.lstm import CustomLSTM

        if batch_norm:
            # if applying batch norm after every LSTM layer, 1-layer LSTMs are required
            self.lstms = torch.nn.ModuleList([])
            self.batch_norms = torch.nn.ModuleList([])
            self.dropouts = torch.nn.ModuleList([]) if dropout else None
            for layer in range(num_layers):
                if layer == 0:
                    layer_input_size = input_size
                else:
                    layer_input_size = hidden_size
                if kwargs["custom_lstm"]:
                    self.lstms.append(
                        CustomLSTM(
                            input_size=layer_input_size,
                            hidden_size=hidden_size,
                            quantize=kwargs["quantize"],
                            rw_dropout=kwargs["rw_dropout"],
                        )
                    )
                else:
                    self.lstms.append(
                        torch.nn.LSTM(
                            input_size=layer_input_size, hidden_size=hidden_size
                        )
                    )
                self.batch_norms.append(torch.nn.BatchNorm1d(num_features=hidden_size))
                if dropout:
                    self.dropouts.append(torch.nn.Dropout(dropout))
        else:
            # if not applying batch norm multi-layer LSTMs can  be used
            if kwargs["custom_lstm"]:
                self.lstm = CustomLSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    quantize=kwargs["quantize"],
                    rw_dropout=kwargs["rw_dropout"],
                )
            else:
                self.lstm = torch.nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            self.dropout = torch.nn.Dropout(dropout) if dropout else None

        for name, v in self.named_parameters():
            if "weight" in name or "bias" in name:
                v.data *= float(weights_init_scale)

        if forget_gate_bias is not None:
            for name, v in self.named_parameters():
                if "bias_ih" in name:
                    v.data[hidden_size : 2 * hidden_size].fill_(forget_gate_bias)
                if "bias_hh" in name:
                    v.data[hidden_size : 2 * hidden_size] *= float(
                        hidden_hidden_bias_scale
                    )

        self.using_custom_lstm = kwargs["custom_lstm"]

    def forward(self, x, h=None):
        # x is (seq_len, batch_size, input_size)
        # h is (h_0, c_0) where both h_0 and c_0 are (num_layers, batch, hidden_size)
        if self.batch_norm:
            # if doing batch_norm apply multiple 1-layer LSTMs
            h_fl = []
            c_fl = []
            for layer in range(self.num_layers):
                if h is None:
                    layer_hidden_states = None
                else:
                    h_0, c_0 = h[0][layer].unsqueeze(0), h[1][layer].unsqueeze(0)
                    # both h_0 and c_0 are (1, batch, hidden_size)
                    layer_hidden_states = (h_0, c_0)
                x, (h_f, c_f), *_ = self.lstms[layer](x, layer_hidden_states)
                # both h_f and c_f are (1, batch, hidden_size)
                # apply batch norm before dropout (which may not be needed at all now)
                # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout # noqa: E501
                # permute x into (N,C,L) for batch norm, and back again afterwards
                x = x.permute(1, 2, 0)
                x = self.batch_norms[layer](x)
                x = x.permute(2, 0, 1)
                # apply dropout after every LSTM layer
                if self.dropouts:
                    x = self.dropouts[layer](x)
                # append layer's output states as (batch, hidden_size) slices
                h_fl.append(h_f[0])
                c_fl.append(c_f[0])
            # compute the final state tensors
            h_f = torch.stack(h_fl, dim=0)  # (num_layers, batch, hidden_size)
            c_f = torch.stack(c_fl, dim=0)  # (num_layers, batch, hidden_size)
            h = (h_f, c_f)
            all_h = None
        else:
            # if not doing batch_norm call the multi-layer LSTM and apply final dropout
            if self.using_custom_lstm:
                x, h, all_h = self.lstm(x, h)
            else:
                x, h = self.lstm(x, h)
                all_h = None
            if self.dropout:
                x = self.dropout(x)

        return x, h, all_h
