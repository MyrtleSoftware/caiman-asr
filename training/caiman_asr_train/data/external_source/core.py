#!/usr/bin/env python3


import numpy as np
from jaxtyping import Int


def str_to_numpy_unicode(s: str) -> Int[np.ndarray, "length"]:  # noqa
    return np.array([ord(char) for char in s], dtype=np.int32)
