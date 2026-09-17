"""Packed-cell helpers the tile_transducer tests build their inputs with."""

import torch
from beartype import beartype
from torch import Tensor as Ten

from caiman_asr_train.core.packing import packed_batch_offsets
from caiman_asr_train.tile_transducer.utils import cell_coords


@beartype
def offsets(f_len: Ten, y_len: Ten) -> tuple[Ten, Ten, int]:
    boff = packed_batch_offsets(f_len, y_len + 1)
    total = int(boff[-1].item())
    Xoff = torch.zeros(len(f_len), dtype=torch.int64, device=f_len.device)
    Xoff[1:] = boff[:-1]
    return boff, Xoff, total


@beartype
def pack_A(
    Fenc: Ten,
    Gpred: Ten,
    f_len: Ten,
    y_len: Ten,
    Xoff: Ten,
    total: int,
    dtype: torch.dtype,
) -> Ten:
    b, t, u = cell_coords(f_len, y_len, Xoff, total)
    return torch.relu(Fenc[b, t] + Gpred[b, u]).to(dtype)
