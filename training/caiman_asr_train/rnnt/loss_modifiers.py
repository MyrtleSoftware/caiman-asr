from dataclasses import dataclass

from beartype import beartype


@beartype
@dataclass
class LossModifiers:
    dp_lam: float
    eos_lam: float
    star_penalty: float


IDENTITY_LOSS_MODIFIERS = LossModifiers(
    dp_lam=0.0,
    eos_lam=0.0,
    star_penalty=1.0,
)
