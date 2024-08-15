#!/usr/bin/env python3

from argparse import ArgumentParser

from beartype import beartype

from caiman_asr_train.train_utils.distributed import print_once


@beartype
def add_delay_penalty_args(parser: ArgumentParser) -> None:
    dp = parser.add_argument_group("Delay penalty arguments")
    dp.add_argument(
        "--delay_penalty",
        default="linear_schedule",
        help="Delay penalty value to control emission latency. Can be a fixed float value "
        "which is kept constant during training, or a `linear_schedule`. ",
    )
    dp.add_argument(
        "--dp_warmup_steps",
        type=int,
        default=5000,
        help="Number of training steps at the start when penalty=`--dp_warmup_penalty`"
        "Valid only if `--delay_penalty` is `linear_schedule`. ",
    )
    dp.add_argument(
        "--dp_warmup_penalty",
        type=float,
        default=0.0,
        help="Penalty value during warmup steps. Recommended to set to 0.0 to "
        "stabilize training. Valid only if `--delay_penalty` is `linear_schedule` ",
    )
    dp.add_argument(
        "--dp_ramp_penalty",
        type=float,
        default=0.007,
        help="Ramp up penalty value at step = dp_warmup_steps + 1. The actual value then "
        "increases linearly until it reaches `--dp_final_penalty`. "
        "Valid only if `--delay_penalty` is `linear_schedule` ",
    )
    dp.add_argument(
        "--dp_final_steps",
        type=int,
        default=20000,
        help="The penalty keeps increasing until here, and then "
        "stays constant past this point. Valid only if `--delay_penalty` is "
        "`linear_schedule`.",
    )
    dp.add_argument(
        "--dp_final_penalty",
        type=float,
        default=0.01,
        help="The final penalty value past the `--dp_final_steps` point. "
        "Valid only if `--dp_delay_penalty` is `linear_schedule`.",
    )


def verify_delay_penalty_args(args):
    dp = args.delay_penalty
    schedule_args = [
        "dp_warmup_steps",
        "dp_warmup_penalty",
        "dp_ramp_penalty",
        "dp_final_steps",
        "dp_final_penalty",
    ]
    if dp == "linear_schedule":
        for arg in schedule_args:
            if getattr(args, arg) is None:
                raise ValueError(
                    f"Argument `{arg}` must be set "
                    "because you set `--delay_penalty linear_schedule`"
                )

        assert (
            args.dp_warmup_steps >= 0
        ), "Number of warmup_steps must be greater or equal to 0"

        assert (
            args.dp_final_steps > args.dp_warmup_steps
        ), "Number of final steps must be greater than number of warmup steps"
    else:
        try:
            float(dp)
        except ValueError:
            print(f"Error: Delay penalty {dp} must be a float/int or 'linear_schedule'")
            raise
        for arg in schedule_args:
            if getattr(args, arg) is not None:
                print_once(
                    f"Argument `--{arg}` set but `--delay_penalty={dp}`, "
                    f"setting {arg} to None"
                )
                setattr(args, arg, None)
