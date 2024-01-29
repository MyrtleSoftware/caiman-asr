#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace

from beartype import beartype


@beartype
def add_shared_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--prob_val_narrowband",
        type=float,
        default=0.0,
        help="Probability that a batch of validation audio gets downsampled to 8kHz and then upsampled to original sample rate",
    )
    parser.add_argument(
        "--inspect_audio",
        action="store_true",
        help="""Save audios (after augmentations are applied) to /results/augmented_audios.
        This will slow down DALI""",
    )
    parser.add_argument(
        "--n_utterances_only",
        default=None,
        type=int,
        help="Abridge the dataset to only this many utterances, selected randomly",
    )


@beartype
def check_shared_args(args: Namespace) -> None:
    if args.read_from_tar:
        assert (
            args.n_utterances_only is None
        ), "n_utterances_only is not supported when reading from tar files"
