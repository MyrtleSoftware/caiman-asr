import torch
from beartype import beartype
from beartype.typing import Optional
from torch import Tensor as Ten

NO_CLAMP = 1e30


@beartype
def cell_coords(f_len: Ten, y_len: Ten, Xoff: Ten, total: int) -> tuple[Ten, Ten, Ten]:
    """The (b, t, u) each packed cell came from."""
    device = f_len.device
    g_len = y_len.to(torch.int64) + 1
    b = torch.repeat_interleave(
        torch.arange(len(f_len), device=device), f_len.to(torch.int64) * g_len
    )
    k = torch.arange(total, device=device) - Xoff.to(torch.int64)[b]
    return b, k // g_len[b], k % g_len[b]


@beartype
def build_cell_index(
    f_len: Ten, y_len: Ten, Xoff: Ten, max_flen: int, max_glen: int, total: int
) -> Ten:
    """Build a flat index of the rectangular (t,u) cells
    in the packed (b,t,u) array."""
    b, t, u = cell_coords(f_len, y_len, Xoff, total)
    return b * (max_flen * max_glen) + t * max_glen + u


@beartype
def pack_scalar(rect: Ten, idx_flat: Ten) -> Ten:
    """Pack a rectangular (b,t,u) array into a flat (b,t,u) array"""
    return rect.reshape(-1)[idx_flat]


@beartype
def build_label_cell(Label: Ten, max_flen: int, max_glen: int, idx_flat: Ten):
    """Build a flat index of the label cells in the packed (b,t,u) array."""
    device = Label.device
    label_len = Label.shape[1]
    if label_len == 0:
        return torch.zeros(idx_flat.numel(), device=device, dtype=torch.int32)
    u = torch.arange(max_glen, device=device).clamp(max=label_len - 1)
    lab_cell = Label[:, u]
    lab_cell = lab_cell[:, None, :].expand(Label.shape[0], max_flen, max_glen)
    return pack_scalar(lab_cell.contiguous().int(), idx_flat).int()


@beartype
def _emit_bias(d: Ten, norm: Ten, lam: float, gamma: float) -> Ten:
    """
    One penalty's per-edge emit bias, ``clamp(lam * d / norm, -gamma, gamma)``,
    where norm = T**mu.
    """
    return (lam * d / norm).clamp_(-gamma, gamma)


@beartype
def apply_emit_penalty(
    lex: Ten,
    f_len: Ten,
    Label: Ten,
    dp_lam: float,
    eos_lam: float,
    eos_idx: Optional[int],
    dp_mu: float = 0.0,
    dp_gamma: float = NO_CLAMP,
) -> None:
    """Apply delay and EOS penalties to the lex array in place."""
    if dp_lam == 0.0 and eos_lam == 0.0:
        return

    _, T, U = lex.shape
    t = torch.arange(T, device=lex.device, dtype=lex.dtype)[None, :, None]
    flen = f_len.to(lex.dtype)[:, None, None]
    d = (flen - 1) / 2 - t
    norm = flen**dp_mu

    if dp_lam != 0.0:
        lex.add_(_emit_bias(d, norm, dp_lam, dp_gamma))

    if eos_lam != 0.0:
        assert eos_idx is not None, "eos_lam != 0 requires an eos_idx"
        label_len = Label.shape[1]
        if label_len == 0:
            return
        u_idx = torch.arange(U, device=lex.device).clamp(max=label_len - 1)
        is_eos = (Label[:, u_idx] == eos_idx)[:, None, :]
        lex.add_(_emit_bias(d, norm, eos_lam, dp_gamma) * is_eos)


@beartype
def compute_coeffs(
    alpha: Ten,
    beta: Ten,
    lbx: Ten,
    lex: Ten,
    audio_len: Ten,
    txt_len: Ten,
    loss_grad: Ten,
):
    """Compute the per-cell coefficients for the backward pass."""
    _, Tn, Un = alpha.shape
    device = alpha.device
    logP = beta[:, 0, 0][:, None, None]

    t_idx = torch.arange(Tn, device=device)[None, :, None]
    u_idx = torch.arange(Un, device=device)[None, None, :]
    flen = audio_len[:, None, None]
    glen = (txt_len + 1)[:, None, None]
    valid = (t_idx < flen) & (u_idx < glen)
    is_last_t = t_idx == flen - 1
    is_last_u = u_idx == glen - 1
    is_terminal = is_last_t & is_last_u

    beta_tp1 = torch.cat([beta[:, 1:, :], beta[:, -1:, :]], dim=1)
    beta_up1 = torch.cat([beta[:, :, 1:], beta[:, :, -1:]], dim=2)
    g = loss_grad[:, None, None]

    def safe_exp(expo, keep):
        return torch.exp(torch.where(keep, expo, torch.full_like(expo, -1e30)))

    c_dense = g * safe_exp(alpha + beta - logP, valid)
    c_blank = g * (
        safe_exp(alpha + lbx - logP, valid & is_terminal)
        + safe_exp(alpha + lbx + beta_tp1 - logP, valid & ~is_last_t)
    )
    c_label = g * safe_exp(alpha + lex + beta_up1 - logP, valid & ~is_last_u)
    return c_dense, c_blank, c_label
