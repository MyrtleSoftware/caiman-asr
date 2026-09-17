from functools import cache

import torch
from beartype import beartype

# Custom type alias for Tilelang dtypes
TLDtype = torch.dtype | str


@beartype
def tl_dtype(dtype: torch.dtype) -> str:
    """
    tilelang dtype string ("float16", ...) for a ``torch.dtype``.

    The kernels accept a ``torch.dtype`` directly, but the tilelang autotuner only
    serializes primitive closure cells (int/float/str/bool/None) -- a ``torch.dtype``
    is rejected -- so the tuner closure passes the dtype to ``get_tir`` as this
    string instead.
    """
    return str(dtype).removeprefix("torch.")


@cache
@beartype
def _dtype_itemsize(name: str) -> int:
    return getattr(torch, name).itemsize


@beartype
def dtype_bytes(cfg: dict, key: str) -> int:
    return _dtype_itemsize(cfg[key])


@beartype
def accum_dtype_for(dtype: torch.dtype) -> torch.dtype:
    """Mirrors ``at::acc_type``: fp32 for fp16/bf16/fp32, fp64 only for fp64."""
    return torch.promote_types(dtype, torch.float32)


@beartype
def amp_dtype_for(t: torch.Tensor) -> torch.dtype:
    """
    The dtype autocast would compute ``t``'s matmuls in, for a kernel that subsumes an
    ``nn.Linear`` and so is never rewritten by autocast itself.

    Must be called from ``nn.Module.forward``: ``torch.amp.custom_fwd`` has already
    disabled autocast by the time an autograd Function body runs.
    """
    device = t.device.type
    if torch.is_autocast_enabled(device):
        return torch.get_autocast_dtype(device)
    return t.dtype


@beartype
def to_contig(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Coerce ``t`` to ``dtype`` in packed row-major layout, no-op if already both."""
    if t.dtype == dtype and t.is_contiguous():
        return t
    return t.to(dtype, memory_format=torch.contiguous_format, copy=True)
