"""Cases, backends and one fwd+bwd runner shared by the RNN-T loss tests."""

from dataclasses import dataclass, replace
from functools import cached_property, partial

import pytest
import torch
from beartype.typing import Callable, Optional
from torch import Tensor as Ten

from caiman_asr_train.autotune.support import sm80_supported, tilelang_supported
from caiman_asr_train.core.packing import packed_batch_offsets
from caiman_asr_train.rnnt.loss_modifiers import LossModifiers

DEV = "cuda"


@dataclass(frozen=True)
class Mods:
    delay: float = 0.0
    eos: float = 0.0
    eos_idx: Optional[int] = None
    dp_mu: float = 0.0
    dp_clamp: Optional[float] = None

    @property
    def loss_mods(self) -> LossModifiers:
        return LossModifiers(dp_lam=self.delay, eos_lam=self.eos, star_penalty=1.0)


NO_MODS = Mods()


@dataclass(frozen=True)
class Lengths:
    label: Ten
    f_len: Ten
    y_len: Ten
    blank_idx: int
    T: int
    U: int
    V: int

    @cached_property
    def boff(self) -> Ten:
        return packed_batch_offsets(self.f_len, self.y_len + 1)

    @cached_property
    def total(self) -> int:
        return int(self.boff[-1].item())

    @property
    def max_f_len(self) -> int:
        return int(self.f_len.max().item())

    @property
    def B(self) -> int:
        return self.f_len.shape[0]

    def pack(self, rect: Ten) -> Ten:
        """Drop the don't-care cells of a (B, T, U, C) tensor."""
        return torch.cat(
            [
                rect[b, : int(self.f_len[b]), : int(self.y_len[b]) + 1].reshape(
                    -1, rect.shape[-1]
                )
                for b in range(self.B)
            ]
        )

    def mask(self) -> Ten:
        t = torch.arange(self.T, device=DEV)[None, :, None]
        u = torch.arange(self.U, device=DEV)[None, None, :]
        return (t < self.f_len[:, None, None]) & (u < (self.y_len + 1)[:, None, None])

    def grad_masks(self) -> tuple:
        return (1,)


@dataclass(frozen=True)
class Case(Lengths):
    """Logits for an unfused loss, held in fp64 and cast per run."""

    logits: Ten

    def leaf(self, dtype: torch.dtype, packed: bool) -> Ten:
        x = self.pack(self.logits) if packed else self.logits
        return x.detach().clone().to(dtype).requires_grad_(True)


@dataclass(frozen=True)
class JointCase(Lengths):
    """Encoder / prediction activations and the output projection, in fp64."""

    f: Ten
    g: Ten
    W: Ten
    b: Ten

    @property
    def H(self) -> int:
        return self.W.shape[0]

    def leaves(self, dtype: torch.dtype) -> tuple[Ten, ...]:
        return tuple(
            t.detach().clone().to(dtype).requires_grad_(True)
            for t in (self.f, self.g, self.W, self.b)
        )

    def activation(self, f: Ten, g: Ten, dtype: torch.dtype) -> Ten:
        from apex.contrib.transducer import TransducerJoint

        joint = TransducerJoint(
            pack_output=True, relu=True, dropout=True, dropout_prob=0.0
        )
        return joint(
            f.contiguous().to(dtype),
            g.contiguous().to(dtype),
            self.f_len,
            self.y_len + 1,
            batch_offset=self.boff,
            packed_batch=self.total,
        )

    def grad_masks(self) -> tuple:
        t = torch.arange(self.T, device=DEV)[None, :]
        u = torch.arange(self.U, device=DEV)[None, :]
        return (
            (t < self.f_len[:, None]).unsqueeze(-1),
            (u < (self.y_len + 1)[:, None]).unsqueeze(-1),
            1,
            1,
        )


def _int(*vals) -> Ten:
    return torch.tensor(vals, device=DEV, dtype=torch.int32)


def _lengths(B, T, U, V, seed, eos_idx, uniform, f_len_vals, y_len_vals) -> dict:
    torch.manual_seed(seed)
    blank_idx = V - 1
    label = torch.randint(0, blank_idx, (B, U - 1), device=DEV, dtype=torch.int32)
    f_len = _int(*[T] * B) if uniform else torch.randint(1, T + 1, (B,), device=DEV)
    y_len = _int(*[U - 1] * B) if uniform else torch.randint(0, U, (B,), device=DEV)
    f_len[0], y_len[-1] = T, U - 1
    if f_len_vals is not None:
        f_len = _int(*f_len_vals)
    if y_len_vals is not None:
        y_len = _int(*y_len_vals)
    if eos_idx is not None:
        for b in range(B):
            last = int(y_len[b].item()) - 1
            if last >= 0:
                label[b, last] = eos_idx
    return dict(
        label=label,
        f_len=f_len.int(),
        y_len=y_len.int(),
        blank_idx=blank_idx,
        T=T,
        U=U,
        V=V,
    )


def case(
    B: int = 4,
    T: int = 7,
    U: int = 6,
    V: int = 128,
    seed: int = 0,
    eos_idx=None,
    uniform: bool = False,
    f_len_vals=None,
    y_len_vals=None,
) -> Case:
    """An unfused-loss case; ``uniform`` makes every sequence full length."""
    return Case(
        **_lengths(B, T, U, V, seed, eos_idx, uniform, f_len_vals, y_len_vals),
        logits=torch.randn(B, T, U, V, device=DEV, dtype=torch.float64) * 0.5,
    )


def cells_case(cells: int, V: int = 128, seed: int = 0) -> Case:
    """A case with exactly ``cells`` packed cells, to sit on kernel block edges."""
    fixed = 3 * 3 + 5 * 2
    assert cells > fixed
    tail = cells - fixed - 1
    return case(
        B=3,
        T=5,
        U=tail + 1,
        V=V,
        seed=seed,
        f_len_vals=[3, 5, 1],
        y_len_vals=[2, 1, tail],
    )


def joint_case(
    B: int = 4,
    T: int = 32,
    U: int = 32,
    H: int = 64,
    V: int = 128,
    seed: int = 0,
    eos_idx=None,
    f_len_vals=None,
    y_len_vals=None,
) -> JointCase:
    """A fused joint+loss case"""
    randn = dict(device=DEV, dtype=torch.float64)
    return JointCase(
        **_lengths(B, T, U, V, seed, eos_idx, False, f_len_vals, y_len_vals),
        f=torch.randn(B, T, H, **randn),
        g=torch.randn(B, U, H, **randn),
        W=torch.randn(H, V, **randn) * (H**-0.5),
        b=torch.randn(V, **randn),
    )


@dataclass(frozen=True)
class Backend:
    name: str
    make: Callable
    dtypes: frozenset
    penalty: bool = True
    rect: bool = True
    tilelang: bool = False
    gradcheck: bool = True
    rtol: float = 0.0
    atol: float = 0.0

    def __str__(self) -> str:
        return self.name


def _unfused(build) -> Callable:
    def make(c: Case, dtype: torch.dtype, mods: Mods, packed: bool) -> Callable:
        module = build(c, dtype, mods, packed)

        def call(x: Ten) -> Ten:
            return module(
                x,
                c.label,
                c.f_len,
                c.y_len,
                c.blank_idx,
                eos_idx=mods.eos_idx,
                batch_offset=c.boff if packed else None,
                max_f_len=c.max_f_len if packed else None,
                delay_penalty=mods.delay,
                eos_penalty=mods.eos,
            )

        return call

    return make


def _cuda_module(c, dtype, mods, packed):
    from rnnt_ext.transducer.loss import TransducerLoss

    assert not mods.dp_mu and mods.dp_clamp is None
    return TransducerLoss(packed_input=packed)


def _triton_module(c, dtype, mods, packed):
    from caiman_asr_train.rnnt.triton_loss import TritonTransducerLoss

    return TritonTransducerLoss(
        packed_input=packed, dp_mu=mods.dp_mu, dp_clamp=mods.dp_clamp
    )


BLOCK_V = 32


def _tilelang_module(c, dtype, mods, packed, block_C=64):
    from tests.tile_transducer.kernels import untuned_kernels

    from caiman_asr_train.tile_transducer.autotune import Mode
    from caiman_asr_train.tile_transducer.tilelang_loss import TileLangTransducerLoss

    cfg = dict(block_C=block_C, block_V=BLOCK_V)
    return TileLangTransducerLoss(
        kernels=untuned_kernels(
            Mode.unfused, 0, c.V, c.blank_idx, dtype, fwd_cfg=cfg, bwd_cfg=cfg
        ),
        packed_input=packed,
        dp_mu=mods.dp_mu,
        dp_clamp=mods.dp_clamp,
    )


def _apex_joint(inner_build) -> Callable:
    """Project, then hand the logits to an unfused loss."""

    def make(c: JointCase, dtype: torch.dtype, mods: Mods, packed: bool) -> Callable:
        call = _unfused(inner_build)(c, dtype, mods, True)
        return lambda A, W, b: call((A @ W + b).contiguous())

    return make


def _fused_joint(
    c: JointCase, dtype: torch.dtype, mods: Mods, packed: bool
) -> Callable:
    from tests.tile_transducer.kernels import untuned_kernels

    from caiman_asr_train.tile_transducer.autotune import Mode
    from caiman_asr_train.tile_transducer.tilelang_proj_and_loss import (
        TileLangProjAndLoss,
    )

    module = TileLangProjAndLoss(
        c.blank_idx,
        kernels=untuned_kernels(Mode.fused, c.H, c.V, c.blank_idx, dtype),
        eos_idx=mods.eos_idx,
        dp_mu=mods.dp_mu,
        dp_clamp=mods.dp_clamp,
    )

    def call(A: Ten, W: Ten, b: Ten) -> Ten:
        return module(
            A, W, b, c.label, c.f_len, c.y_len, c.boff, c.total, mods.loss_mods
        )

    return call


ALL_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32, torch.float64})
TRITON_DTYPES = ALL_DTYPES if sm80_supported() else ALL_DTYPES - {torch.bfloat16}
TILELANG_DTYPES = frozenset({torch.float16, torch.float32, torch.float64})
JOINT_DTYPES = frozenset({torch.float16, torch.float32})

CUDA = Backend(
    "cuda", _unfused(_cuda_module), ALL_DTYPES, penalty=False, rtol=1e-4, atol=1e-4
)
TRITON = Backend(
    "triton", _unfused(_triton_module), TRITON_DTYPES, rtol=1e-4, atol=1e-4
)
TILELANG = Backend(
    "tilelang-unfused",
    _unfused(_tilelang_module),
    TILELANG_DTYPES,
    rect=False,
    tilelang=True,
    gradcheck=False,
    rtol=2e-2,
    atol=2e-2,
)

APEX_TRITON = Backend(
    "apex+triton", _apex_joint(_triton_module), JOINT_DTYPES, rect=False
)
APEX_CUDA = Backend(
    "apex+cuda", _apex_joint(_cuda_module), JOINT_DTYPES, penalty=False, rect=False
)
TILELANG_FUSED = Backend(
    "tilelang-fused",
    _fused_joint,
    JOINT_DTYPES,
    rect=False,
    tilelang=True,
    gradcheck=False,
    rtol=2e-2,
    atol=2e-2,
)


def tilelang_blocks(block_C: int) -> Backend:
    return replace(
        TILELANG,
        name=f"tilelang-C{block_C}",
        make=_unfused(partial(_tilelang_module, block_C=block_C)),
    )


LOSS_BACKENDS = [TRITON, CUDA, TILELANG]
JOINT_BACKENDS = [APEX_TRITON, APEX_CUDA, TILELANG_FUSED]


NO_TRITON_BF16 = pytest.mark.skipif(
    not sm80_supported(),
    reason="the Triton loss kernels need compute capability >= 8.0 for bfloat16",
)

NO_TILELANG = pytest.mark.skipif(
    not tilelang_supported(),
    reason="the TileLang kernels need compute capability >= 8.0 and torch >= 2.7",
)


def LAYOUTS(b: "Backend") -> tuple:
    return (True, False) if b.rect else (True,)


def DTYPES(only=None) -> Callable:
    def axis(b: "Backend") -> list:
        return sorted(b.dtypes & frozenset(only or b.dtypes), key=str)

    return axis


def _label(value) -> str:
    if isinstance(value, bool):
        return "packed" if value else "rect"
    return str(value).removeprefix("torch.")


def params(backends: list["Backend"], axis: Optional[Callable] = None) -> list:
    """Parametrize over backends, optionally crossed with a second axis."""
    out = []
    for b in backends:
        marks = [NO_TILELANG] if b.tilelang else []
        values = [None] if axis is None else axis(b)
        for v in values:
            args = (b,) if axis is None else (b, v)
            name = b.name if axis is None else f"{b.name}-{_label(v)}"
            out.append(pytest.param(*args, id=name, marks=marks))
    return out


def with_penalty(backends: list[Backend]) -> list[Backend]:
    return [b for b in backends if b.penalty]


def with_fp64(backends: list[Backend]) -> list[Backend]:
    return [b for b in backends if torch.float64 in b.dtypes]


def with_gradcheck(backends: list[Backend]) -> list[Backend]:
    return [b for b in backends if b.gradcheck]


def others(backends: list[Backend]) -> list[Backend]:
    """Every backend bar the reference, which is the first."""
    return backends[1:]


@dataclass(frozen=True)
class Result:
    loss: Ten
    inputs: tuple

    @property
    def grads(self) -> tuple:
        return tuple(t.grad for t in self.inputs)


def run(
    backend: Backend,
    c: Lengths,
    dtype: torch.dtype = torch.float16,
    mods: Mods = NO_MODS,
    packed: bool = True,
    joint_dtype: Optional[torch.dtype] = None,
    scale: float = 1.0,
) -> Result:
    """Forward and backward one case through one backend."""
    call = backend.make(c, joint_dtype or dtype, mods, packed)
    if isinstance(c, JointCase):
        f, g, W, b = c.leaves(dtype)
        jd = joint_dtype or dtype
        loss = call(c.activation(f, g, jd), W.to(jd).contiguous(), b.to(jd))
        inputs = (f, g, W, b)
    else:
        x = c.leaf(dtype, packed)
        loss = call(x)
        inputs = (x,)
    loss.backward(torch.full_like(loss, scale))
    return Result(loss.detach(), inputs)


def module(
    backend: Backend,
    c: Lengths,
    dtype: torch.dtype = torch.float16,
    mods: Mods = NO_MODS,
    packed: bool = True,
) -> Callable:
    """The bare callable, for tests that own the leaves themselves."""
    return backend.make(c, dtype, mods, packed)


def loss_only(backend: Backend, c: Case, dtype, mods=NO_MODS, packed=True) -> Callable:
    """A gradcheck stub; the clone protects the input from an in-place backward."""
    call = module(backend, c, dtype, mods, packed)
    return lambda x: call(x.clone())


DTYPE_TOL = {
    torch.float16: 1e-2,
    torch.bfloat16: 1e-1,
    torch.float32: 1e-4,
    torch.float64: 1e-10,
}


def assert_close(
    got: Ten, want: Ten, backend: Backend, dtype: torch.dtype = torch.float32, **over
) -> None:
    """Tolerate the worse of the backend's own error and the precision's."""
    floor = DTYPE_TOL[dtype]
    tol = dict(rtol=max(backend.rtol, floor), atol=max(backend.atol, floor)) | over
    torch.testing.assert_close(got.float(), want.float(), **tol)
