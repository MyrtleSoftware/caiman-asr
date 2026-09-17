"""The packed-cell offsets and the flat cell index must stay 64-bit: at the production
vocab ``X`` passes 2**31 elements, so an int32 anywhere wraps negative."""

import torch
from tests.tile_transducer.packing import offsets

from caiman_asr_train.tile_transducer.utils import build_cell_index

CELLS = 2**31  # f_len * g_len per batch element: exactly int32's overflow point


def test_offsets_carry_more_cells_than_int32_holds():
    f_len = torch.full((3,), 2**20, dtype=torch.int32)
    y_len = torch.full((3,), 2**11 - 1, dtype=torch.int32)  # g_len = y_len + 1

    boff, Xoff, total = offsets(f_len, y_len)

    assert boff.dtype is torch.int64 and Xoff.dtype is torch.int64
    assert boff.tolist() == [CELLS, 2 * CELLS, 3 * CELLS]
    assert Xoff.tolist() == [0, CELLS, 2 * CELLS]
    assert total == 3 * CELLS


def test_cell_index_is_int64():
    f_len = torch.tensor([3, 2], dtype=torch.int32)
    y_len = torch.tensor([1, 2], dtype=torch.int32)
    _, Xoff, total = offsets(f_len, y_len)

    idx = build_cell_index(f_len, y_len, Xoff, max_flen=3, max_glen=3, total=total)

    assert idx.dtype is torch.int64
    assert idx.tolist() == [0, 1, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
