import os
import sys
from textwrap import dedent, indent

from beartype import beartype

from caiman_asr_train.train_utils.distributed import print_once


def is_stdout_tty():
    if hasattr(sys.stdout, "isatty"):
        return sys.stdout.isatty()

    try:
        return os.isatty(sys.stdout.fileno())
    except Exception:
        return False


@beartype
def lhotse_warn(msg: str):
    """
    Print a warning with indentation and a yellow "WARNING" prefix.
    """

    warn = "WARNING:"

    if is_stdout_tty():
        warn = f"\033[93m{warn}\033[0m"

    print_once(f"{warn}\n{indent(dedent(msg), prefix='  ')}")
