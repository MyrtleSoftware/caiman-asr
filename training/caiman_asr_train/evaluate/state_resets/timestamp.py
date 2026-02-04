from dataclasses import dataclass

from beartype import beartype
from beartype.typing import TypeAlias, Union


@beartype
@dataclass
class FullStamp:
    """
    A dataclass to represent a timestamp.

    Attributes:
        model: The model timestamp, i.e. the frame that the model predicts
               the audio was spoken at. This is computed from the final
               so can be before or after the first time the partial predicts
               a given word.
        user_perceived: The user perceived timestamp, i.e. the first frame
               that the user observed the final text (after any partial->final
               overwrites).
        final_time: The frame that the final was received at.
    """

    model: int
    user_perceived: int
    final_time: int


Timestamp: TypeAlias = Union[FullStamp, int]


@beartype
def model_time(time: Timestamp) -> int:
    """
    Return the model time of the timestamp.
    """
    match time:
        case FullStamp(model, _, _):
            return model
        case _:
            return time


@beartype
def user_perceived_time(time: Timestamp) -> int:
    """
    Return the user perceived time of the timestamp.
    """
    match time:
        case FullStamp(_, user_perceived, _):
            return user_perceived
        case _:
            return time


@beartype
def final_time(time: Timestamp) -> int:
    """
    Return the final time of the timestamp.
    """
    match time:
        case FullStamp(_, _, final_time):
            return final_time
        case _:
            return time


@beartype
def add_frames(time: Timestamp, n: int) -> Timestamp:
    """
    Add n frames to the timestamp and return a new timestamp.
    """
    match time:
        case FullStamp(model, user_perceived, final_time):
            return FullStamp(model + n, user_perceived + n, final_time + n)
        case _:
            return time + n
