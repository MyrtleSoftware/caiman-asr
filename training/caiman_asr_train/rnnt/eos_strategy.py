from dataclasses import dataclass

from beartype import beartype
from beartype.typing import TypeAlias, Union


@beartype
@dataclass
class EOSIgnore:
    eos_idx: int


@beartype
@dataclass
class EOSBlank:
    eos_idx: int


@beartype
@dataclass
class EOSPredict:
    eos_idx: int
    alpha: float
    beta: float


EOSStrategy: TypeAlias = Union[None, EOSIgnore, EOSPredict, EOSBlank]
