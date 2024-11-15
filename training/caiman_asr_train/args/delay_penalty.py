#!/usr/bin/env python3

from argparse import ArgumentParser

from beartype import beartype


@beartype
def add_delay_penalty_args(parser: ArgumentParser) -> None:
    dp = parser.add_argument_group("Delay penalty arguments")

    dp.add_argument(
        "--delay_penalty",
        default="wer_schedule",
        help="Delay penalty value to control emission latency. Can be a fixed float value "
        "which is kept constant during training, or a 'wer_schedule' which scales with "
        "the dev 'wer'. ",
    )

    dp.add_argument(
        "--dp_initial_value",
        type=float,
        default=0.0,
        help="Initial value of the delay penalty.",
    )
    dp.add_argument(
        "--dp_final_value",
        type=float,
        default=0.01,
        help="Value of the penalty after toggle_step.",
    )
    dp.add_argument(
        "--dp_toggle_step",
        type=int,
        help="Fallback step at which the penalty is set to final_value.",
    )
    dp.add_argument(
        "--dp_wer_threshold",
        type=float,
        default=0.3,
        help="If WER is below this value, the penalty is set to final_value.",
    )


def verify_delay_penalty_args(args):
    #
    dp = args.delay_penalty

    if dp == "wer_schedule":
        assert args.dp_initial_value is not None
        assert args.dp_final_value is not None
        assert args.dp_wer_threshold is not None or args.dp_toggle_step is not None
    else:
        try:
            float(dp)
        except ValueError:
            print(f"Error: Delay penalty {dp} must be a float/int or 'wer_schedule'")
            raise
