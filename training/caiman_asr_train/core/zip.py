import builtins
from typing import overload

import torch
from beartype import beartype
from beartype.typing import Union

from caiman_asr_train.core.lstm import Hidden
from caiman_asr_train.core.model import StateEncoder, StatePred


@beartype
def _zip(xs: list[Hidden]) -> Hidden:
    """
    Concatenate batch-size-1 hidden states into a single batched Hidden.
    """

    assert len(xs) > 0, "Need at least one hidden state to zip"

    assert all(
        x.cell.shape == xs[0].cell.shape for x in xs
    ), "Hidden states must have the same shape"
    assert all(
        x.hidd.shape == xs[0].hidd.shape for x in xs
    ), "Hidden states must have the same shape"

    assert all(x.cell.shape[1] == 1 for x in xs), "Hidden states must have batch size 1"
    assert all(x.hidd.shape[1] == 1 for x in xs), "Hidden states must have batch size 1"

    return Hidden(
        hidd=torch.cat([x.hidd for x in xs], dim=1),
        cell=torch.cat([x.cell for x in xs], dim=1),
    )


@beartype
def _unzip(x: Hidden) -> list[Hidden]:
    """
    Split a batched Hidden into a list of batch-size-1 Hidden states.

    Inverse of _zip().
    """

    assert x.hidd.shape == x.cell.shape, "Hidden/cell shape mismatch"

    B = x.hidd.shape[1]

    return [
        Hidden(hidd=x.hidd[:, i : i + 1], cell=x.cell[:, i : i + 1]) for i in range(B)
    ]


@overload
def zip(states: list[StateEncoder]) -> StateEncoder: ...


@overload
def zip(states: list[StatePred]) -> StatePred: ...


@beartype
def zip(
    states: Union[list[StateEncoder], list[StatePred]],
) -> Union[StateEncoder, StatePred]:
    """
    Zip a list of batch-size-1 states into a single batched state.
    """

    assert len(states) > 0, "Need at least one state to zip"

    match states[0]:
        case StateEncoder():
            return StateEncoder(
                pre_stack=_zip([s.pre_stack for s in states]),
                aft_stack=_zip([s.aft_stack for s in states]),
            )
        case StatePred():
            assert all(
                s.embed.shape == states[0].embed.shape for s in states
            ), "Embeddings must have the same shape"
            assert all(
                s.embed.shape[1] == 1 for s in states
            ), "Embeddings must have batch size 1"

            return StatePred(
                state=_zip([s.state for s in states]),
                embed=torch.cat([s.embed for s in states], dim=1),
            )


@overload
def unzip(state: StateEncoder) -> list[StateEncoder]: ...


@overload
def unzip(state: StatePred) -> list[StatePred]: ...


@beartype
def unzip(
    state: Union[StateEncoder, StatePred],
) -> Union[list[StateEncoder], list[StatePred]]:
    """
    Split a batched state into a list of batch-size-1 states.

    Inverse of zip().
    """

    match state:
        case StateEncoder():
            return [
                StateEncoder(pre_stack=pre, aft_stack=aft)
                for pre, aft in builtins.zip(
                    _unzip(state.pre_stack), _unzip(state.aft_stack), strict=True
                )
            ]
        case StatePred():
            assert (
                state.embed.shape[1] == state.state.hidd.shape[1]
            ), "Embedding/state batch mismatch"

            return [
                StatePred(state=h, embed=state.embed[:, i : i + 1])
                for i, h in enumerate(_unzip(state.state))
            ]
