from dataclasses import dataclass

import torch
from beartype import beartype
from beartype.typing import Optional, TypeAlias, Union
from torch import Tensor

from caiman_asr_train.autotune.support import (
    MIN_TORCH_FOR_TILELANG,
    tilelang_supported,
)
from caiman_asr_train.core.pytree import register_dataclass_as_pytree


@dataclass
@beartype
class LegacyConfig:
    quantize: bool
    rw_dropout: Optional[float]


@dataclass
class CustomConfig:
    rw_dropout: Optional[float]


@dataclass
class TorchConfig:
    # When True, forward() returns the per-timestep AllHidden by stepping the
    # nn.LSTM one timestep at a time (nn.LSTM only exposes the final state).
    return_all: bool


@dataclass
class TileConfig:
    pass


BackendConfig: TypeAlias = Union[CustomConfig, LegacyConfig, TorchConfig, TileConfig]


@dataclass
@beartype
class ConfigLSTM:
    # Shared between all backends
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    # State dict should be backend independent
    backend: BackendConfig
    # Initialisation
    weights_init_scale: Optional[float]
    forget_gate_bias: Optional[float]
    hidden_hidden_bias_scale: Optional[float]


@register_dataclass_as_pytree
@beartype
@dataclass
class Hidden:
    """
    LSTM final hidden/cell states L=layers, B=batch, H=hidden.

    Members:
        hidd: (L, B, H)
        cell: (L, B, H)
    """

    hidd: torch.Tensor
    cell: torch.Tensor

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hidd, self.cell

    def detach(self) -> "Hidden":
        return Hidden(self.hidd.detach(), self.cell.detach())


@beartype
@dataclass
class AllHidden:
    """
    LSTM all hidden/cell states L=layers, T=time, B=batch, H=hidden.

    Members:
        hidd: (L, T, B, H)
        cell: (L, T, B, H)
    """

    hidd: torch.Tensor
    cell: torch.Tensor

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hidd, self.cell


class LSTM(torch.nn.Module):
    @beartype
    def __init__(self, config: ConfigLSTM):
        """Returns an LSTM with forget gate bias init to `config.forget_gate_bias`.

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
        super().__init__()

        self.config = config

        match config.backend:
            case LegacyConfig():
                # Legacy CustomLSTM import takes O(30s) on startup due to third-party
                # qtorch import. As such, only import when needed
                from rnnt_ext.custom_lstm.legacy import CustomLSTM as LegacyLSTM

                self.lstm = LegacyLSTM(
                    input_size=config.input_size,
                    hidden_size=config.hidden_size,
                    num_layers=config.num_layers,
                    dropout=config.dropout,
                    quantize=config.backend.quantize,
                    rw_dropout=(
                        0.0
                        if config.backend.rw_dropout is None
                        else config.backend.rw_dropout
                    ),
                )

            case CustomConfig():
                assert torch.cuda.is_available(), "Custom LSTM requires CUDA"

                from rnnt_ext.custom_lstm.lstm import CustomLSTM

                self.lstm = CustomLSTM(
                    input_size=config.input_size,
                    hidden_size=config.hidden_size,
                    num_layers=config.num_layers,
                    dropout=config.dropout,
                    rw_dropout=(
                        0.0
                        if config.backend.rw_dropout is None
                        else config.backend.rw_dropout
                    ),
                )

            case TileConfig():
                # TileLangLSTM pulls in tilelang (a heavy, CUDA-only dependency) at
                # import time, so -- like the Custom/Legacy backends above -- only
                # import it when this backend is actually selected.
                from caiman_asr_train.tilelstm import TileLangLSTM

                assert tilelang_supported(), (
                    "TileLang LSTM requires a GPU with compute capability >= SM80+ "
                    f"and torch >= {MIN_TORCH_FOR_TILELANG}"
                )

                self.lstm = TileLangLSTM(
                    input_size=config.input_size,
                    hidden_size=config.hidden_size,
                    num_layers=config.num_layers,
                    dropout=config.dropout,
                )

            case TorchConfig():
                self.lstm = torch.nn.LSTM(
                    input_size=config.input_size,
                    hidden_size=config.hidden_size,
                    num_layers=config.num_layers,
                    dropout=config.dropout,
                )

        self.dropout = torch.nn.Dropout(config.dropout)

        hidden = config.hidden_size

        w_scale = config.weights_init_scale
        f_scale = config.forget_gate_bias
        h_scale = config.hidden_hidden_bias_scale

        for name, v in self.named_parameters():
            if "weight" in name or "bias" in name:
                if w_scale is not None:
                    v.data *= w_scale
            if "bias_ih" in name:
                if f_scale is not None:
                    v.data[hidden : 2 * hidden].fill_(f_scale)
            if "bias_hh" in name:
                if h_scale is not None:
                    v.data[hidden : 2 * hidden] *= h_scale

    @beartype
    def forward(
        self, x: Tensor, h: Optional[Hidden] = None
    ) -> tuple[Tensor, Hidden, Optional[AllHidden]]:

        maybe_h = None if h is None else h.as_tuple()

        if isinstance(self.config.backend, TorchConfig):
            if self.config.backend.return_all:
                # nn.LSTM only exposes the final (h_n, c_n); feed one timestep
                # at a time, threading state, to collect every layer's
                # hidden/cell state at each timestep (matching the custom
                # backend's AllHidden).
                hidd, cell = [], []
                for t in range(x.shape[0]):
                    _, maybe_h = self.lstm(x[t : t + 1], maybe_h)
                    hidd.append(maybe_h[0])  # (L, B, H)
                    cell.append(maybe_h[1])  # (L, B, H)
                all_h = AllHidden(torch.stack(hidd, dim=1), torch.stack(cell, dim=1))
                # Last layer's per-timestep hidden == nn.LSTM's output sequence.
                x = all_h.hidd[-1]
            else:
                x, maybe_h = self.lstm(x, maybe_h)
                all_h = None
        else:
            x, maybe_h, maybe_all_h = self.lstm(x, maybe_h)
            all_h = None if maybe_all_h is None else AllHidden(*maybe_all_h)

        if self.config.dropout:
            x = self.dropout(x)

        # Wrap in strong types

        return x, Hidden(*maybe_h), all_h
