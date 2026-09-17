#!/usr/bin/env python3

from argparse import ArgumentParser

from beartype import beartype


@beartype
def add_noise_augmentation_args(parser: ArgumentParser) -> None:
    noise = parser.add_argument_group("Noise augmentation")
    noise.add_argument(
        "--prob_background_noise",
        default=0.25,
        type=float,
        help="Probability of applying background noise augmentation",
    )

    noise.add_argument(
        "--noise_snr_low",
        type=float,
        default=10.0,
        help="Low end of the signal-to-noise-ratio range for noise augmentation (DB)",
    )
    noise.add_argument(
        "--noise_snr_high",
        type=float,
        default=20.0,
        help="High end of the signal-to-noise-ratio range for noise augmentation (DB)",
    )

    noise.add_argument(
        "--noise_dataset",
        default="Myrtle/CAIMAN-ASR-BackgroundNoise",
        type=str,
        help="The name of an audio dataset on the Hugging Face Hub",
    )
    noise.add_argument(
        "--noise_config",
        default="train",
        type=str,
        help="""When loading a noise dataset from the Hugging Face Hub,
        this option allows you to specify the configuration if needed""",
    )

    noise.add_argument(
        "--prob_room_aug",
        default=0.0,
        type=float,
        help="Probability of applying room augmentation",
    )
    noise.add_argument(
        "--room_dataset",
        type=str,
        help="Path to the noise dataset for room augmentation",
    )
    noise.add_argument(
        "--rir_early_only",
        action="store_true",
        help="""When set, only the early reflections (first 50 ms) will be used,
        This speeds up the convolution process.""",
    )
