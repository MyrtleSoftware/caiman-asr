"""
Config cross-product helpers for building a kernel's candidate tile sets.
"""

from collections.abc import Iterator
from itertools import product

from beartype import beartype


@beartype
def divisors(n: int) -> list[int]:
    """
    All positive divisors of ``n``, ascending.
    """
    return [d for d in range(1, n + 1) if n % d == 0]


@beartype
def config_product(base: dict, **axes: list) -> Iterator[dict]:
    """
    The cartesian product of the named tile ``axes``, each combination merged onto ``base``.
    """
    keys = list(axes)
    for combo in product(*axes.values()):
        yield {**base, **dict(zip(keys, combo))}
