"""The module-level TileLang gate the TileLang kernel tests share."""

import pytest

from caiman_asr_train.autotune.support import (
    MIN_TORCH_FOR_TILELANG,
    tilelang_supported,
)


def skip_without_tilelang() -> None:
    if not tilelang_supported():
        pytest.skip(
            f"needs a CUDA GPU of compute capability >= 8.0 (Ampere+) and torch "
            f">= {MIN_TORCH_FOR_TILELANG}; below that the TileLang kernels emit "
            "instructions the device lacks, or the autotuner cannot read "
            "the device properties it filters configs on",
            allow_module_level=True,
        )
