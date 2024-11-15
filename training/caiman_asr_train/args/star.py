import argparse

from beartype import beartype


@beartype
def add_star_args(parser: argparse.ArgumentParser) -> None:
    star = parser.add_argument_group("Star arguments")

    star.add_argument(
        "--star_initial_value",
        type=float,
        default=0.75,
        help="Initial value of the star penalty.",
    )

    star.add_argument(
        "--star_final_value",
        type=float,
        default=1.0,
        help="Value of the penalty after toggle_step.",
    )

    star.add_argument(
        "--star_toggle_step",
        type=int,
        help="Step at which the penalty is set to final_value.",
    )

    star.add_argument(
        "--star_wer_threshold",
        type=float,
        default=0.2,
        help="If WER is below this value, the penalty is set to final_value.",
    )
