"""No kernel may write past the packed cell dimension. compute-sanitizer cannot see
TileLang OOB, so sentinel tails are the only bounds check we have."""

import pytest
import torch
from tests.sm80 import skip_without_tilelang

skip_without_tilelang()

from tests.tile_transducer.packing import offsets  # noqa: E402

from caiman_asr_train.tile_transducer.autotune.candidates import BLOCK_CS  # noqa: E402
from caiman_asr_train.tile_transducer.autotune.configs import (  # noqa: E402
    logsumexp_configs,
    proj_and_loss_bwd_configs,
    proj_lse_configs,
    transducer_loss_bwd_configs,
)
from caiman_asr_train.tile_transducer.tilelang_kernels.tilelang_lse import (  # noqa: E402
    logsumexp,
    proj_lse,
)
from caiman_asr_train.tile_transducer.tilelang_kernels.tilelang_transducer import (  # noqa: E402,E501
    proj_and_loss_bwd,
    transducer_loss_bwd,
)
from caiman_asr_train.tile_transducer.utils import (  # noqa: E402
    build_cell_index,
    build_label_cell,
)

H, V, BLANK = 48, 100, 99
DEV = "cuda"
PAD_ROWS = 512
SENTINEL = 1.25
FIXED_CELLS = 19  # f_len [3, 5, 1] x g_len [3, 2, *]


def padded(rows, cols, dtype):
    shape = (rows + PAD_ROWS,) if cols is None else (rows + PAD_ROWS, cols)
    big = torch.full(shape, SENTINEL, device=DEV, dtype=dtype)
    return big, big[:rows]


def assert_tails_untouched(*buffers):
    for big, rows in buffers:
        assert (big[rows:] == SENTINEL).all()


def cell_batch(cells, dtype):
    torch.manual_seed(0)
    assert cells > FIXED_CELLS
    f_len = torch.tensor([3, 5, 1], device=DEV, dtype=torch.int32)
    y_len = torch.tensor([2, 1, cells - FIXED_CELLS - 1], device=DEV, dtype=torch.int32)
    _, Xoff, total = offsets(f_len, y_len)
    assert total == cells

    max_flen, max_glen = 5, int(y_len.max().item()) + 1
    label = torch.randint(0, BLANK, (3, max_glen - 1), device=DEV, dtype=torch.int32)
    idx_flat = build_cell_index(f_len, y_len, Xoff, max_flen, max_glen, total)
    return dict(
        cells=cells,
        n_rect=3 * max_flen * max_glen,
        cell_idx=idx_flat,
        label_cell=build_label_cell(label, max_flen, max_glen, idx_flat),
        A=torch.randn(cells, H, device=DEV, dtype=dtype),
        W=torch.randn(H, V, device=DEV, dtype=dtype) * (H**-0.5),
        Bias=torch.randn(V, device=DEV, dtype=dtype),
        logits=torch.randn(cells, V, device=DEV, dtype=dtype),
        c_dense=torch.randn(cells, device=DEV, dtype=torch.float32),
        c_blank=torch.randn(cells, device=DEV, dtype=torch.float32),
        c_label=torch.randn(cells, device=DEV, dtype=torch.float32),
        denom=torch.full((cells,), 3.0, device=DEV, dtype=torch.float32),
    )


@pytest.mark.parametrize(
    "dtype, block_H, block_V_bwd, tol",
    [(torch.float16, 64, 64, 2e-2), (torch.float32, 32, 32, 1e-4)],
    ids=["fp16", "fp32"],
)
def test_all_four_kernels_stay_inside_the_cell_dimension(
    dtype, block_H, block_V_bwd, tol
):
    block_C = 64
    b = cell_batch(2 * block_C + 5, dtype)  # a ragged trailing block
    cells, accum = b["cells"], torch.float32
    dts = dict(dtype=dtype, accum_dtype=accum)

    denom_big, denom = padded(cells, None, accum)
    lbx_big, lbx = padded(b["n_rect"], None, accum)
    lex_big, lex = padded(b["n_rect"], None, accum)
    x_big, x = padded(cells, V, dtype)

    proj_lse(H, V, BLANK, block_C=block_C, block_V=64, block_H=block_H, **dts)(
        b["A"], b["W"], b["Bias"], b["label_cell"], b["cell_idx"], denom, lbx, lex, x
    )
    logsumexp(V, BLANK, block_C=block_C, block_V=64, **dts)(
        b["logits"], b["label_cell"], b["cell_idx"], denom, lbx, lex
    )

    dx_big, dx = padded(cells, V, dtype)
    dx.copy_(b["logits"])
    transducer_loss_bwd(V, BLANK, block_C=block_C, block_V=64, **dts)(
        b["c_dense"], b["c_blank"], b["c_label"], b["denom"], b["label_cell"], dx
    )

    fx_big, fx = padded(cells, V, dtype)
    fx.copy_(b["logits"])
    da_big, dA = padded(cells, H, dtype)
    db = torch.zeros(V, device=DEV, dtype=accum)
    proj_and_loss_bwd(
        H, V, BLANK, block_C=block_C, block_V=block_V_bwd, block_H=block_H, **dts
    )(
        b["c_dense"],
        b["c_blank"],
        b["c_label"],
        b["denom"],
        b["label_cell"],
        fx,
        b["W"],
        dA,
        db,
    )
    torch.cuda.synchronize()

    assert_tails_untouched(
        (denom_big, cells),
        (lbx_big, b["n_rect"]),
        (lex_big, b["n_rect"]),
        (x_big, cells),
        (dx_big, cells),
        (fx_big, cells),
        (da_big, cells),
    )
    for got in (denom, x, dx, fx, dA):
        assert torch.isfinite(got).all()
    # db reduces over cells, so it also covers the ragged trailing block's masking.
    torch.testing.assert_close(db, fx.float().sum(dim=0), rtol=tol, atol=tol)


def test_transducer_loss_bwd_scales_past_the_grid_y_cap():
    """The cell blocks must ride grid.x: ``gridDim.y`` is capped at 65535, so mapping
    them to grid.y died with CUDA_ERROR_INVALID_VALUE above ``65535 * block_C`` cells --
    2.1M at the smallest tuned ``block_C=32``, inside what a real batch reaches."""
    block_C = min(BLOCK_CS)
    cells = block_C * 65536 + 5  # one block past the cap, and ragged
    need = cells * (V * 2 + 5 * 4)
    if torch.cuda.mem_get_info()[0] < 2 * need:
        pytest.skip(f"needs ~{2 * need / 2**30:.1f} GiB free")

    torch.manual_seed(0)
    x_big, x = padded(cells, V, torch.float16)
    x.normal_()
    lab = torch.randint(0, V, (cells,), device=DEV, dtype=torch.int32)
    cd, cb, cl = (torch.rand(cells, device=DEV, dtype=torch.float32) for _ in range(3))
    dn = torch.full((cells,), 3.0, device=DEV, dtype=torch.float32)

    tail = slice(cells - 8, cells)  # the cells only the past-the-cap blocks cover
    x_in = x[tail].clone()

    transducer_loss_bwd(V, BLANK, block_C=block_C, block_V=64)(cd, cb, cl, dn, lab, x)
    torch.cuda.synchronize()

    want = cd[tail].unsqueeze(1) * torch.exp(x_in.float() - dn[tail].unsqueeze(1))
    want[:, BLANK] -= cb[tail]
    want.scatter_add_(1, lab[tail].long().unsqueeze(1), -cl[tail].unsqueeze(1))
    torch.testing.assert_close(x[tail].float(), want, rtol=2e-2, atol=2e-2)
    assert_tails_untouched((x_big, cells))


def cheapest_per_block_c(configs):
    seen = {}
    for c in sorted(configs, key=lambda c: (c["block_C"] * c["block_V"], c["threads"])):
        seen.setdefault(c["block_C"], c)
    return list(seen.values())


@pytest.mark.parametrize(
    "make_configs,factory",
    [
        (proj_lse_configs, proj_lse),
        (proj_and_loss_bwd_configs, proj_and_loss_bwd),
        (logsumexp_configs, logsumexp),
        (transducer_loss_bwd_configs, transducer_loss_bwd),
    ],
    ids=["proj_lse", "proj_and_loss_bwd", "logsumexp", "transducer_loss_bwd"],
)
def test_every_candidate_block_c_compiles(make_configs, factory):
    """A candidate that fails to elaborate is silently dropped from the tuner's search
    space, so compile the cheapest candidate at each block_C here instead."""
    configs = make_configs(H, V, BLANK, torch.device(DEV))
    assert configs
    for cfg in cheapest_per_block_c(configs):
        factory.compile(**cfg)
