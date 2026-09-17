from collections.abc import Callable

import torch
from beartype import beartype

from caiman_asr_train.utils.math import ceil_div


@beartype
def _zero_band_counter(
    M: int, block_M: int, fpad: int, device: torch.device
) -> torch.Tensor:
    """
    A freshly-zero'd per-band arrival counter ``F`` ``[cdiv(M, block_M),
    fpad]`` int32, allocated per call.
    """
    return torch.zeros(ceil_div(M, by=block_M), fpad, device=device, dtype=torch.int32)


@beartype
def wrap_splitk_bwd(kernel: object, cfg: dict) -> Callable:
    """
    Wrap the 7-arg split-K backward ``(As, Cs, R, delta, dGates, P, dC)``
    behind the coop backward's 5-arg call contract.
    """
    k_splits = cfg["k_splits"]

    def call(As, Cs, R, delta, dGates):
        M, H = As.shape[1], As.shape[3]
        P = torch.empty(k_splits, M, H, device=As.device, dtype=torch.float32)
        dC = torch.empty(M, H, device=As.device, dtype=torch.float32)
        return kernel(As, Cs, R, delta, dGates, P, dC)

    return call


@beartype
def wrap_coop_fwd_flags(kernel: object, cfg: dict) -> Callable:
    """
    Wrap a band-flag forward (``lstm_coop_fwd`` OR ``lstm_persistent_fwd`` -- both drive
    their per-step sync with the band counter ``F``) behind the layer's 6-arg forward
    call contract ``(Y0, C0, R, gates, Ys, Cs)``.
    """
    block_M, fpad = cfg["block_M"], cfg["fpad"]
    needs_cpp = "blocks_per_sm" in cfg  # persistent forward carries the fp32 ping-pong

    def call(Y0, C0, R, gates, Ys, Cs):
        M, H = Y0.shape
        F = _zero_band_counter(M, block_M, fpad, Y0.device)
        if needs_cpp:
            Cpp = torch.empty(2, M, H, device=Y0.device, dtype=torch.float32)
            return kernel(Y0, C0, R, gates, Ys, Cs, F, Cpp)
        return kernel(Y0, C0, R, gates, Ys, Cs, F)

    return call


@beartype
def wrap_coop_bwd_flags(kernel: object, cfg: dict) -> Callable:
    """
    Wrap a band-flag backward (``lstm_coop_bwd`` OR ``lstm_persistent_bwd`` -- both drive
    their per-step sync with the band counter ``F``) behind the 5-arg call contract,
    supplying the per-call ``F`` (see ``_zero_band_counter``).
    """
    block_M, fpad = cfg["block_M"], cfg["fpad"]
    needs_dc = "blocks_per_sm" in cfg  # persistent backward carries the dc_fut scratch

    def call(As, Cs, R, delta, dGates):
        M = As.shape[1]
        F = _zero_band_counter(M, block_M, fpad, As.device)
        if needs_dc:
            # fp32 (the kernel's dtype_hi carry), NOT Cs.dtype -- Cs is now fp16.
            dC = torch.empty(M, As.shape[3], device=As.device, dtype=torch.float32)
            return kernel(As, Cs, R, delta, dGates, dC, F)
        return kernel(As, Cs, R, delta, dGates, F)

    return call
