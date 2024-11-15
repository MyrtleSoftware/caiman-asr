from dataclasses import dataclass

from beartype import beartype
from beartype.typing import TypeAlias, Union


@beartype
@dataclass
class FullStamp:
    """
    A dataclass to represent a timestamp.

    Attributes:
        model: The model timestamp, i.e. frame emitted by the model.
               Note that this can be overly optimistic. i.e. if the model
               emits a partial at time t, changes its mind and
               emits a different partial at time t+1, and then finalizes
               the first partial at time t+2, the model timestamp will
               still be time t
        user_perceived: The user perceived timestamp, i.e. frame after
            any partial->final overwrites.
    """

    model: int
    user_perceived: int


Timestamp: TypeAlias = Union[FullStamp, int]


@beartype
def model_time(time: Timestamp) -> int:
    """
    Return the model time of the timestamp.
    """
    match time:
        case FullStamp(model, _):
            return model
        case _:
            return time


@beartype
def user_perceived_time(time: Timestamp) -> int:
    """
    Return the user perceived time of the timestamp.
    """
    match time:
        case FullStamp(_, user_perceived):
            return user_perceived
        case _:
            return time


@beartype
def add_frames(time: Timestamp, n: int) -> Timestamp:
    """
    Add n frames to the timestamp and return a new timestamp.
    """
    match time:
        case FullStamp(model, user_perceived):
            return FullStamp(model + n, user_perceived + n)
        case _:
            return time + n
