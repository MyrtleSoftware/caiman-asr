"""TileLang kernel plumbing: the forward against torch, tile configs and JIT caching."""

import pytest
import torch
from tests.rnnt.loss.framework import DEV, JointCase, joint_case, run
from tests.sm80 import skip_without_tilelang

skip_without_tilelang()

from tests.tile_transducer.kernels import untuned_kernels  # noqa: E402
from tests.tile_transducer.packing import offsets, pack_A  # noqa: E402

from caiman_asr_train.autotune.dtypes import accum_dtype_for  # noqa: E402
from caiman_asr_train.rnnt.loss_modifiers import (  # noqa: E402
    IDENTITY_LOSS_MODIFIERS,
)
from caiman_asr_train.tile_transducer.autotune import Mode  # noqa: E402
from caiman_asr_train.tile_transducer.tilelang_kernels import (  # noqa: E402
    tilelang_transducer as tl_bwd,
)
from caiman_asr_train.tile_transducer.tilelang_kernels.tilelang_lse import (  # noqa: E402
    logsumexp,
    proj_lse,
    proj_lse_run,
    ref_torch,
)
from caiman_asr_train.tile_transducer.tilelang_proj_and_loss import (  # noqa: E402
    TileLangProjAndLoss,
)
from caiman_asr_train.tile_transducer.utils import (  # noqa: E402
    build_cell_index,
    build_label_cell,
    pack_scalar,
)


def unfused_forward_run(V: int, blank_idx: int, cfg: dict | None = None):
    """The unfused forward wearing the fused forward's signature."""
    kernel = logsumexp(V, blank_idx, **(cfg or {}))

    def run(A, W, b, label_cell, cell_idx, denom, lbx, lex, X, total):
        torch.addmm(b, A, W, beta=1.0, alpha=1.0, out=X)
        kernel(X, label_cell, cell_idx, denom, lbx.view(-1), lex.view(-1))

    return run


def forward_outputs(c: JointCase, dtype: torch.dtype, fwd_run, cfg=None) -> dict:
    accum = accum_dtype_for(dtype)
    _, Xoff, total = offsets(c.f_len, c.y_len)
    idx = build_cell_index(c.f_len, c.y_len, Xoff, c.T, c.U, total)
    out = dict(
        denom=torch.zeros(total, device=DEV, dtype=accum),
        lbx=torch.zeros(c.B, c.T, c.U, device=DEV, dtype=accum),
        lex=torch.zeros(c.B, c.T, c.U, device=DEV, dtype=accum),
        X=torch.zeros(total, c.V, device=DEV, dtype=dtype),
    )
    fwd_run(
        pack_A(c.f, c.g, c.f_len, c.y_len, Xoff, total, dtype=dtype),
        c.W.to(dtype),
        c.b.to(dtype),
        build_label_cell(c.label, c.T, c.U, idx),
        idx,
        out["denom"],
        out["lbx"],
        out["lex"],
        out["X"],
        total,
    )
    return out | dict(idx=idx, total=total)


def torch_reference(c: JointCase) -> dict:
    A = torch.relu(c.f[:, :, None, :] + c.g[:, None, :, :]).float()
    denom, lbx, lex = ref_torch(A, c.W.float(), c.b.float(), c.label, c.blank_idx)
    X = torch.einsum("btuh,hk->btuk", A, c.W.float()) + c.b.float()
    return dict(denom=denom, lbx=lbx, lex=lex, X=X)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("fwd", ["fused", "unfused"])
def test_forward_matches_the_torch_reference(dtype, fwd):
    """denom / lbx / lex / packed X against fp32 torch, for both forward kernels."""
    c = joint_case(H=48, V=100)
    cfg = dict(block_C=64, block_V=64) | (dict(block_H=32) if fwd == "fused" else {})
    kw = dict(dtype=dtype, accum_dtype=accum_dtype_for(dtype), **cfg)
    fwd_run = (
        proj_lse_run(proj_lse(c.H, c.V, c.blank_idx, **kw))
        if fwd == "fused"
        else unfused_forward_run(c.V, c.blank_idx, cfg=kw)
    )
    got = forward_outputs(c, dtype, fwd_run)
    want = torch_reference(c)
    mask = c.mask()
    mask_lex = mask & (
        torch.arange(c.U, device=DEV)[None, None, :] < c.y_len[:, None, None]
    )

    assert got["X"].dtype is dtype and got["denom"].dtype is accum_dtype_for(dtype)
    torch.testing.assert_close(
        got["denom"],
        pack_scalar(want["denom"], got["idx"]).to(got["denom"].dtype),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        got["lbx"][mask], want["lbx"][mask].to(got["lbx"].dtype), rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        got["lex"][mask_lex],
        want["lex"][mask_lex].to(got["lex"].dtype),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        got["X"].float(),
        c.pack(want["X"]).float(),
        rtol=2e-2,
        atol=2e-2,
    )


def test_a_mismatched_dtype_is_refused():
    """The kernels are built for one precision at setup, so a mismatch must fail."""
    c = joint_case(B=2)
    f, g, W, b = c.leaves(torch.float32)
    module = TileLangProjAndLoss(
        c.blank_idx,
        kernels=untuned_kernels(Mode.fused, c.H, c.V, c.blank_idx, torch.float16),
    )
    with pytest.raises(AssertionError, match="tuned for"):
        module(
            c.activation(f, g, torch.float32),
            W,
            b,
            c.label,
            c.f_len,
            c.y_len,
            c.boff,
            c.total,
            IDENTITY_LOSS_MODIFIERS,
        )


def test_tuned_cfgs_reach_the_kernels(monkeypatch):
    """A dropped ``cfg=`` is otherwise silent: the kernels fall back to their own
    defaults, so production loses the tuning with no error and no wrong answer."""
    import tests.tile_transducer.kernels as kernels_mod

    seen = {}

    def spy(stage):
        def factory(H, V, blank_idx, **cfg):
            seen[stage] = cfg
            return lambda *args: None

        return factory

    monkeypatch.setattr(kernels_mod, "proj_lse", spy("fwd"))
    monkeypatch.setattr(kernels_mod, "proj_and_loss_bwd", spy("bwd"))
    fwd_cfg = dict(block_C=64, block_V=48, block_H=64, num_stages=2, threads=128)
    bwd_cfg = dict(block_C=32, block_V=96, block_H=32, num_stages=3, threads=256)
    untuned_kernels(Mode.fused, 64, 128, 127, torch.float16, fwd_cfg, bwd_cfg)

    assert fwd_cfg.items() <= seen["fwd"].items()
    assert bwd_cfg.items() <= seen["bwd"].items()


def test_dynamic_shapes_need_no_recompile():
    """batch / max_flen / max_glen are runtime inputs, so the cache is hit again."""
    from tests.rnnt.loss.framework import TILELANG_FUSED

    assert proj_lse(64, 128, blank_idx=127) is proj_lse(64, 128, blank_idx=127)
    assert tl_bwd.transducer_loss_fwd() is tl_bwd.transducer_loss_fwd()
    for B, T, U in [(2, 32, 16), (1, 16, 48)]:
        got = run(TILELANG_FUSED, joint_case(B=B, T=T, U=U))
        assert torch.isfinite(got.loss).all()
        for grad in got.grads:
            assert torch.isfinite(grad).all()
