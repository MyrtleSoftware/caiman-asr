import os
import re
import sys
from argparse import ArgumentParser

from beartype import beartype

from caiman_asr_train.evaluate.error_rates import ErrorRate
from caiman_asr_train.evaluate.metrics import word_error_rate


class MyrtleArgumentParser(ArgumentParser):

    def error(self, message):
        # Copy the default error message format:
        eprint(self.format_usage())
        program_name = os.path.basename(sys.argv[0])
        eprint(f"{program_name}: error: {message}")

        match = re.search(r"unrecognized arguments: (.+)", message)
        if match:
            unrecognized_args = match.group(1).split()

            # This gets the first possible name for each flag,
            # but it may not handle positional arguments correctly
            valid_args = [action.option_strings[0] for action in self._actions]

            for unrecognized_arg in unrecognized_args:
                close_matches = [
                    valid_arg
                    for valid_arg in valid_args
                    if is_close(unrecognized_arg, valid_arg)
                ]
                if close_matches:
                    eprint(
                        f"Did you mean one of these for '{unrecognized_arg}'? "
                        f"{', '.join(close_matches)}"
                    )

        exit(2)


@beartype
def is_close(x: str, y: str) -> bool:
    cer, _, _ = word_error_rate([x], [y], error_rate=ErrorRate.CHAR, standardize=False)
    return cer < 0.5


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
