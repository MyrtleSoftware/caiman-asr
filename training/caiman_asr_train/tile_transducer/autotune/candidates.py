"""
Candidate tile cross-products for the TileLang transducer autotuner.

One generator per kernel. Each yields the raw tile cross-product merged with the
caller's ``base`` kwargs (H, V, blank_idx, dtype / accum_dtype).
"""

from collections.abc import Iterator

from beartype import beartype

from caiman_asr_train.autotune.candidates import config_product

BLOCK_CS = [32, 64, 96, 128, 192, 256]


@beartype
def proj_lse_candidate_tiles(base: dict) -> Iterator[dict]:
    yield from config_product(
        base,
        block_C=BLOCK_CS,
        block_V=[32, 64, 128, 256],
        block_H=[32, 64, 128],
        num_stages=[2, 3],
        threads=[128, 256],
    )


@beartype
def logsumexp_candidate_tiles(base: dict) -> Iterator[dict]:
    yield from config_product(
        base,
        block_C=BLOCK_CS,
        block_V=[64, 128, 256, 512],
        num_stages=[2, 3],
        threads=[128, 256],
    )


@beartype
def proj_and_loss_bwd_candidate_tiles(base: dict) -> Iterator[dict]:
    yield from config_product(
        base,
        block_C=BLOCK_CS,
        block_V=[32, 64, 128],
        block_H=[32, 64, 128, 256],
        num_stages=[1, 2, 3],
        threads=[128, 256],
    )


@beartype
def transducer_loss_bwd_candidate_tiles(base: dict) -> Iterator[dict]:
    yield from config_product(
        base,
        block_C=BLOCK_CS,
        block_V=[64, 128, 256, 512],
        threads=[128, 256],
    )
