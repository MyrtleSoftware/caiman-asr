#!/usr/bin/env python3

import os
from pathlib import Path

from beartype import beartype


@beartype
def pretty_path(path: Path) -> Path:
    """Changes /root/path to /home/user/path"""
    if (home := os.environ.get("USER_HOME")) is None:
        # Give up because user's home is unknown
        return path
    parts = path.parts
    if parts[:2] != ("/", "root"):
        # Give up because path isn't as expected
        return path
    return Path(home) / Path(*parts[2:])
