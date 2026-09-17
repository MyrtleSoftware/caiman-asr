import torch
from beartype import beartype


@beartype
def scan_buffers(
    H: int, B: int, S: int, lo: torch.dtype, hi: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, ...]:
    """
    Fresh kernel inputs in the requested dtypes. R is scaled by 1/H so the
    low-precision recurrence stays bounded; the gate axis is explicit (R [4,H,H],
    gates [S,B,4,H], gate-major). Ys/Cs are NaN-filled so an unwritten slot is
    caught rather than read as stale zeros. ``Cs`` is the saved-cell OUTPUT buffer,
    stored fp16 (``lo``) -- the kernel keeps the running cell in fp32 internally; only
    the cell SEED ``C0`` is fp32 (``hi``).
    """
    Y = torch.randn(B, H, device=device, dtype=lo)
    C0 = torch.randn(B, H, device=device, dtype=hi)
    R = torch.randn(4, H, H, device=device, dtype=lo) / H
    gates = torch.randn(S, B, 4, H, device=device, dtype=lo)
    Ys = torch.full((S + 1, B, H), float("nan"), device=device, dtype=lo)
    Cs = torch.full((S + 1, B, H), float("nan"), device=device, dtype=lo)
    return Y, C0, R, gates, Ys, Cs


@beartype
def bwd_buffers(
    H: int, B: int, S: int, lo: torch.dtype, hi: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, ...]:
    """
    Fresh ``lstm_coop_bwd`` inputs + its (S+1)-slot output buffer.

    ``As`` holds post-activation gates, built as sigmoid/tanh of random
    pre-activations (realistic ranges keep fp16 well-conditioned; the kernel
    computes the same formula for any inputs). ``R`` is scaled by 1/H. ``dGates``
    is zeroed so the pad slot S reads as the first reversed step's recurrent term.
    ``Cs`` (the saved cell the backward reads) is fp16 (``lo``) -- the kernel up-casts
    each read to fp32 for the gradient math. Returns (As, Cs, R, delta, dGates).
    """
    pre = torch.randn(S, B, 4, H, device=device, dtype=hi)
    As = torch.empty(S, B, 4, H, device=device, dtype=lo)
    As[:, :, 0] = torch.sigmoid(pre[:, :, 0])
    As[:, :, 1] = torch.sigmoid(pre[:, :, 1])
    As[:, :, 2] = torch.tanh(pre[:, :, 2])
    As[:, :, 3] = torch.sigmoid(pre[:, :, 3])
    Cs = torch.randn(S + 1, B, H, device=device, dtype=lo)
    R = torch.randn(4, H, H, device=device, dtype=lo) / H
    delta = torch.randn(S, B, H, device=device, dtype=lo)
    dGates = torch.zeros(S + 1, B, 4, H, device=device, dtype=lo)
    return As, Cs, R, delta, dGates
