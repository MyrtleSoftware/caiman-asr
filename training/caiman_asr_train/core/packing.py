"""
Packed-cell offsets shared by the apex and tilelang transducer paths.
"""

import torch
from beartype import beartype


@beartype
def packed_batch_offsets(f_len: torch.Tensor, g_len: torch.Tensor) -> torch.Tensor:
    """
    Cumulative packed-cell offsets ``cumsum(f_len * g_len)``, formed in 64-bit.
    """
    return torch.cumsum(f_len.to(torch.int64) * g_len.to(torch.int64), dim=0)
