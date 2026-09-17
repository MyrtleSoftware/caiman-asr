"""
Autotune + dispatch for the TileLang LSTM kernels.

The public API is the three selection entry points. Each tunes the recurrent-scan
backends for the shape, benchmarks the tuned winners at the batch, and returns the
faster compiled kernel:

* ``select_fwd_kernel`` -- inference forward.
* ``select_fwd_train_kernel`` -- gate-saving training forward.
* ``select_bwd_train_kernel`` -- recurrent backward.
"""

from caiman_asr_train.tilelstm.autotune.select import (
    select_bwd_train_kernel,
    select_fwd_kernel,
    select_fwd_train_kernel,
)

__all__ = [
    "select_fwd_kernel",
    "select_fwd_train_kernel",
    "select_bwd_train_kernel",
]
