from dataclasses import dataclass

from beartype import beartype
from beartype.typing import TypeAlias, Union
from lhotse import Seconds


@beartype
@dataclass
class FiltStats:
    """
    Store the global statistics of a single dataset after filtering
    on duration and characters. This is information is printed
    during startup of the data loader if it is not user-supplied.
    """

    tot_utterances: int
    min_duration: float
    max_duration: float
    sum_duration: float
    squ_duration: float


@beartype
@dataclass
class DurationStats:
    """
    Statistics about utterances.
    """

    min_duration: Seconds
    max_duration: Seconds

    avg_duration: Seconds
    var_duration: float  # Really Seconds^2


@beartype
@dataclass
class FiniteStats(DurationStats):
    """
    If a pipeline is finite then this is it's total duration and
    number of utterances.
    """

    total_utt: int
    total_dur: Seconds


@beartype
@dataclass
class InfiniteStats(DurationStats):
    """
    If a pipeline is infinite then each manifest will repeat/orbit
    on average at a given rate. This class contains the min/max
    orbits in the dataset.

    Stats are floating point as they are expected values.
    """

    min_orbit_utt: float
    max_orbit_utt: float

    min_orbit_dur: Seconds
    max_orbit_dur: Seconds


EpochStats: TypeAlias = Union[FiniteStats, InfiniteStats]
