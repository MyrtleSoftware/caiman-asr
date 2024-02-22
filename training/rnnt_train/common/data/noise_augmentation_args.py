#!/usr/bin/env python3

from argparse import ArgumentParser
from dataclasses import dataclass

from beartype import beartype
from beartype.typing import Optional


@beartype
@dataclass
class NoiseAugmentationArgs:
    noise_dataset: Optional[str]
    noise_config: Optional[str]
    use_noise_audio_folder: bool
    prob_background_noise: float
    prob_babble_noise: float


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
        "--prob_babble_noise",
        default=0.0,
        type=float,
        help="Probability of applying babble noise augmentation",
    )
    noise.add_argument(
        "--noise_delay_steps",
        type=int,
        default=4896,
        help="number of steps delay before noise augmentation starts to ramp up",
    )
    noise.add_argument(
        "--noise_ramp_steps",
        type=int,
        default=4896,
        help="number of steps over which noise augmentation ramps up",
    )
    noise.add_argument(
        "--noise_initial_low",
        type=int,
        default=30,
        help="initial low end of the signal-to-noise-ratio range for noise "
        "augmentation, in dB",
    )
    noise.add_argument(
        "--noise_initial_high",
        type=int,
        default=60,
        help="initial high end of the signal-to-noise-ratio range for noise "
        "augmentation, in dB",
    )
    noise.add_argument(
        "--noise_dataset",
        default="Myrtle/CAIMAN-ASR-BackgroundNoise",
        type=str,
        help="""Either:
        - The name of an audio dataset on the Hugging Face Hub
        - The path to a local noise dataset,
            in which case you should also pass --use_noise_audio_folder
        """,
    )
    noise.add_argument(
        "--noise_config",
        default=None,
        type=str,
        help="""When loading a noise dataset from the Hugging Face Hub,
        this option allows you to specify the configuration if needed""",
    )
    noise.add_argument(
        "--use_noise_audio_folder",
        action="store_true",
        help="Use Hugging Face's AudioFolder format to read a noise dataset. "
        "Set this when reading a local noise dataset",
    )
