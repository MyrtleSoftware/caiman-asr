#!/usr/bin/env python3

import torch
from beartype import beartype
from beartype.typing import Callable

from caiman_asr_train.core.model import Hidden, StateEncoder, StatePred
from caiman_asr_train.rnnt.model import StateRNNT
from caiman_asr_train.utils.math import ceil_div


@beartype
def _map(state: StateRNNT, fn: Callable[[torch.Tensor], torch.Tensor]) -> StateRNNT:
    @beartype
    def map_hidden(part: Hidden) -> Hidden:
        return Hidden(
            hidd=fn(part.hidd),
            cell=fn(part.cell),
        )

    return StateRNNT(
        enc=StateEncoder(
            pre_stack=map_hidden(state.enc.pre_stack),
            aft_stack=map_hidden(state.enc.aft_stack),
        ),
        pred=StatePred(
            state=map_hidden(state.pred.state),
            embed=fn(state.pred.embed),
        ),
    )


@beartype
def _scale_down_state(batch_size: int, state: StateRNNT) -> StateRNNT:
    """
    Scale down the RNNT state to the given batch size.

    Args:
        batch_size (int): The new batch size.
        state (StateRNNT): The original RNNT state.

    Returns:
        StateRNNT: The scaled down RNNT state.
    """

    return _map(state, lambda x: x[:, :batch_size, :])


@beartype
def _scale_up_state(batch_size: int, state: StateRNNT) -> StateRNNT:
    """
    Scale up the RNNT state to the given batch size by repeating the existing states.

    Args:
        batch_size (int): The new batch size.
        state (StateRNNT): The original RNNT state.

    Returns:
        StateRNNT: The scaled up RNNT state.
    """

    n = ceil_div(batch_size, by=state.pred.embed.shape[1])

    return _map(state, lambda x: x.repeat(1, n, 1)[:, :batch_size, :])


@beartype
def scale_state(batch_size: int, state: StateRNNT) -> StateRNNT:
    """
    Scale the RNNT state to the given batch size, this will involve
    truncating or repeating the previous state.

    Args:
        batch_size (int): The new batch size.
        state (StateRNNT): The original RNNT state.

    Returns:
        StateRNNT: The scaled RNNT state.
    """

    current_batch_size = state.pred.embed.shape[1]

    if batch_size == current_batch_size:
        return state
    elif batch_size < current_batch_size:
        return _scale_down_state(batch_size, state)
    else:
        return _scale_up_state(batch_size, state)
