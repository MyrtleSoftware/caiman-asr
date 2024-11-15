import argparse

from beartype import beartype


@beartype
def add_eos_shared_args(parser: argparse.ArgumentParser) -> None:
    eos = parser.add_argument_group("EOS shared arguments")

    eos.add_argument(
        "--eos_is_terminal",
        action="store_true",
        help="Terminate decoding after the first EOS token.",
    )

    eos.add_argument(
        "--eos_vad_threshold",
        type=float,
        default=float("inf"),
        help="Silence threshold (seconds for decoder termination).",
    )

    eos.add_argument(
        "--eos_decoding",
        type=str,
        default="predict",
        choices=["ignore", "blank", "predict", "none"],
        help="How to handle the EOS token during decoding.",
    )

    eos.add_argument(
        "--eos_alpha",
        type=float,
        default=1.0,
        help="The alpha bias applied to the EOS token during decoding.",
    )
    eos.add_argument(
        "--eos_beta",
        type=float,
        default=0.0,
        help="The acceptance threshold for the EOS token during decoding.",
    )


@beartype
def add_eos_train_args(parser: argparse.ArgumentParser) -> None:
    eos = parser.add_argument_group("EOS train-only arguments")

    eos.add_argument(
        "--eos_penalty",
        type=float,
        default=0.0,
        help="Equivalent to delay penalty, but only applied "
        "to the EOS token during training.",
    )
