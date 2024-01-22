#!/usr/bin/env python3

from dataclasses import dataclass

import torch
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Float, Int, jaxtyped


@jaxtyped
@beartype
@dataclass
class EncoderState:
    pre_rnn: Tuple[
        Float[torch.Tensor, "pre_rnn_layers batch hidden_size"],
        Float[torch.Tensor, "pre_rnn_layers batch hidden_size"],
    ]
    post_rnn: Tuple[
        Float[torch.Tensor, "post_rnn_layers batch hidden_size"],
        Float[torch.Tensor, "post_rnn_layers batch hidden_size"],
    ]


@jaxtyped
@beartype
@dataclass
class PredNetState:
    next_to_last_pred_state: Tuple[
        Float[torch.Tensor, "layers batch hidden_size"],
        Float[torch.Tensor, "layers batch hidden_size"],
    ]
    last_token: Int[torch.Tensor, "batch 1"]


@jaxtyped
@beartype
@dataclass
class RNNTState:
    enc_state: EncoderState
    pred_net_state: PredNetState
