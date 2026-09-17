"""
Shared base-2 math helpers for the tilelang RNN-T kernels.
"""

import math

import tilelang.language as T

LOG2E = math.log2(math.e)
LOGE2 = math.log(2.0)
NEG = -1.0e30


@T.macro
def exp(x):
    """exp(x), via the fast base-2 intrinsic."""
    return T.exp2(x * LOG2E)


@T.macro
def log(x):
    """log(x), via the fast base-2 intrinsic."""
    return T.log2(x) * LOGE2
