"""
Candidate tile cross-products for the TileLang LSTM autotuner.

One generator per axis-set (fwd / bwd / split-K). Each yields the raw tile
cross-product merged with the caller's ``base`` kwargs (H, dtype_lo, dtype_hi,
save_gates / k_splits ...); the feasibility filters + derived-field maps live in
``configs.py``.
"""

from collections.abc import Iterator

from beartype import beartype

from caiman_asr_train.autotune.candidates import config_product, divisors
from caiman_asr_train.autotune.feasibility import ATOM_M, ATOM_N


@beartype
def fwd_candidate_tiles(hidden: int, base: dict) -> Iterator[dict]:
    """
    The forward tile cross-product (block_M/N/K x num_stages x threads), each
    merged with ``base`` (the caller's kernel kwargs: H, dtype_lo, dtype_hi,
    save_gates, ...).
    """
    yield from config_product(
        base,
        num_stages=list(range(1, 6)),
        threads=[128, 256],
        block_M=list(range(64, 257, ATOM_M)),
        block_N=list(range(8, 65, ATOM_N // 4)),  # MMA width is 4 times on fwd
        block_K=[d for d in divisors(hidden) if 32 <= d <= 256],
    )


@beartype
def bwd_candidate_tiles(hidden: int, base: dict) -> Iterator[dict]:
    """
    The backward tile cross-product (the lever-D region), each merged with ``base``
    (the caller's kernel kwargs).
    """
    yield from config_product(
        base,
        num_stages=list(range(2, 8)),
        threads=[128, 256],
        block_M=list(range(32, 129, ATOM_M)),
        block_N=list(range(32, 193, ATOM_N)),
        block_K=[d for d in divisors(hidden) if 64 <= d <= 256],
    )


@beartype
def splitk_candidate_tiles(hidden: int, base: dict) -> Iterator[dict]:
    """
    The split-K candidate cross-product over ``k_splits`` x block_N/M/K x num_stages x
    threads x red_M, each merged with ``base`` (H, dtype_lo, dtype_hi).
    """
    yield from config_product(
        base,
        num_stages=list(range(2, 5)),
        threads=[128, 256],
        block_N=list(range(32, 193, ATOM_N)),
        block_M=list(range(16, 257, ATOM_M)),
        block_K=[d for d in divisors(hidden) if 64 <= d <= 256],
        k_splits=[d for d in divisors(4 * hidden) if 2 <= d <= 32],
        red_M=[4, 8, 16],
    )
