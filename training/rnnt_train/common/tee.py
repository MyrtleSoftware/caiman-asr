#!/usr/bin/env python3
import sys
from pathlib import Path

from beartype import beartype


class Tee:
    """Writing a stream to a tee object will
    send it to each file in `files`.
    Based on https://stackoverflow.com/a/11325249"""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


@beartype
def start_logging_stdout_and_stderr(output_dir: str, timestamp: str, name: str) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = open(out_dir / f"{name}_log_{timestamp}.txt", "a")
    sys.stdout = Tee(sys.stdout, f)
    sys.stderr = Tee(sys.stderr, f)
