from dataclasses import dataclass

import torch
from beartype import beartype


@beartype
@dataclass(frozen=True)
class DeviceLimits:
    """
    The handful of device properties the config filters reason about.
    """

    sms: int  # streaming multiprocessors (cooperative grid must co-reside)
    smem_per_block: int  # max opt-in shared memory per block (bytes)
    smem_per_sm: int  # shared memory per SM (bytes) -> blocks/SM by SMEM
    threads_per_sm: int  # -> blocks/SM by thread occupancy
    regs_per_sm: int  # 32-bit registers per SM -> blocks/SM by register pressure
    max_blocks_per_sm: int = 32  # hardware cap (Ampere/Hopper = 32)


@beartype
def device_limits(device: torch.device) -> DeviceLimits:
    """
    Read the ``DeviceLimits`` for ``device`` off ``torch.cuda``.
    """
    prop = torch.cuda.get_device_properties(device)

    return DeviceLimits(
        sms=prop.multi_processor_count,
        smem_per_block=prop.shared_memory_per_block_optin,
        smem_per_sm=prop.shared_memory_per_multiprocessor,
        threads_per_sm=prop.max_threads_per_multi_processor,
        regs_per_sm=prop.regs_per_multiprocessor,
    )


@beartype
def device_name(device: torch.device) -> str:
    """
    Stable per-GPU-model identifier for the selection cache key.
    """
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return device.type


@beartype
def resident_blocks(
    lim: DeviceLimits, *, smem_per_block: int, threads: int, regs_per_block: int
) -> int:
    """
    How many blocks of this config can be simultaneously resident device-wide.

    Args:
        ``regs_per_block`` the caller's per-block register estimate.
    """
    by_smem = lim.smem_per_sm // smem_per_block
    by_threads = lim.threads_per_sm // threads
    by_regs = lim.regs_per_sm // regs_per_block

    # If SMEM fits one block, at least one co-resides; the register estimate only
    # *reduces* occupancy below the SMEM/thread bound, never below 1.
    per_sm = min(by_smem, max(1, by_regs), by_threads, lim.max_blocks_per_sm)

    return lim.sms * max(0, per_sm)
