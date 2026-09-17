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
        default=0.02,
        help="Value of the penalty after toggle_step. (default: %(default)s)",
    )
    dp.add_argument(
        "--dp_wer_threshold_final",
        type=float,
        default=0.4,
        help="If WER is below this value, the penalty is set to final_value.",
    )
    dp.add_argument(
        "--dp_wer_threshold_initial",
        type=float,
        default=0.95,
        help="If WER is above this value, the penalty is set to initial_value.",
    )
    dp.add_argument(
        "--dp_mu",
        type=float,
        default=0.13,
        help="Exponent mu in [0, 1] for the T**mu normalization of the per-edge "
        "delay penalty: bias = clamp(lam*((T-1)/2 - t)/T**mu, -dp_clamp, dp_clamp). "
        "(default: %(default)s)",
    )
    dp.add_argument(
        "--dp_clamp",
        type=float,
        default=None,
        help="Clamp the per-edge delay (and eos) penalty bias to +/- this; "
        "unset (default None) disables the clamp",
    )


def verify_delay_penalty_args(args):
    #
    dp = args.delay_penalty

    if dp == "wer_schedule":
        assert args.dp_initial_value is not None
        assert args.dp_final_value is not None
        assert args.dp_wer_threshold_initial is not None
        assert args.dp_wer_threshold_final is not None
        assert args.dp_wer_threshold_initial != args.dp_wer_threshold_final
    else:
        try:
            float(dp)
        except ValueError:
            print(f"Error: Delay penalty {dp} must be a float/int or 'wer_schedule'")
            raise

    assert 0.0 <= args.dp_mu <= 1.0, f"--dp_mu must be in [0, 1], got {args.dp_mu}"
    if args.dp_clamp is not None:
        assert args.dp_clamp > 0.0, (
            f"--dp_clamp must be > 0 when set, got {args.dp_clamp}; omit it "
            "(None) to disable the clamp"
        )
