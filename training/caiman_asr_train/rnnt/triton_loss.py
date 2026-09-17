# Implement the RNN-T loss using Triton
from dataclasses import dataclass

import rnnt_ext.cuda.logsumexp as logsumexp_cu
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from beartype import beartype
from beartype.typing import List, Optional

# Sentinel dp_gamma meaning "no clamp": far above any real per-edge bias
NO_CLAMP = 1e30


@beartype
@dataclass
class PenaltyParams:
    dp_lam: float
    eos_lam: float
    eos_idx: int
    dp_mu: float = 0.0
    dp_gamma: float = NO_CLAMP


_TORCH_TO_TRITON_DTYPE = {
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}


@triton.jit
def logaddexp(a, b):
    # libdevice.log1p == std::log1p<float> (hardware __log1p instruction).
    # Two corner cases matter here:
    # - Both inputs -inf (zero-probability paths merging, e.g. after an
    #   overflowed logit blocks every route into a lattice cell): the naive
    #   form computes (-inf) - (-inf) = NaN; the true result is -inf.
    # - A NaN input (sub_or_nan poisoning after a +inf logit) must propagate,
    #   as it does through the CUDA log_sum_exp's `>=` branch. tl.maximum/
    #   tl.minimum silently drop NaN operands, which would launder a poisoned
    #   path into a finite-garbage log-prob.
    m = tl.maximum(a, b)
    d = tl.minimum(a, b) - m
    r = tl.where(m == float("-inf"), float("-inf"), m + libdevice.log1p(tl.exp(d)))
    either_nan = (libdevice.isnan(a) != 0) | (libdevice.isnan(b) != 0)
    return tl.where(either_nan, float("nan"), r)


@triton.jit
def sub_or_nan(num, den):
    # Compute num - den, or NaN when den is non-finite (mirrors CUDA sub_or_nan).
    # This is used to compute log(a/b) = log(a) - log(b), handles b=0.
    # A +inf denom (fp16 logit overflow in the logsumexp) would otherwise give
    # finite - inf = -inf: a silently wrong "zero probability" log-prob.
    den_finite = (libdevice.isinf(den) == 0) & (libdevice.isnan(den) == 0)
    return tl.where(den_finite, num - den, float("nan"))


@triton.jit
def _log_softmax_x(
    my_x_ptr, my_denom_ptr, t, u, k, my_stride, dict_size, ACC_DTYPE: tl.constexpr
):
    # x[t, u, k] - denom[t, u]  (t, u, k may be scalars or vectors)
    x_off = (t * my_stride + u).to(tl.int64) * dict_size + k
    # Cast x to acc dtype to match denom (mirrors CUDA Scalar -> Acc promotion)
    x_val = tl.load(my_x_ptr + x_off).to(ACC_DTYPE)
    denom_val = tl.load(my_denom_ptr + t * my_stride + u)  # already acc dtype
    return sub_or_nan(x_val, denom_val)


@triton.jit
def frac_penalty(lam, t, T, mu, gamma):
    """
    Bias added to emit edges (a reward -- positive early, negative late):

      clamp(lam * ((T - 1) / 2 - t) / T**mu, -gamma, gamma).
    """
    raw = lam * ((T - 1) / 2 - t) / libdevice.pow(T.to(tl.float32), mu)
    return tl.minimum(tl.maximum(raw, -gamma), gamma)


@triton.jit
def log_null_x(
    my_x_ptr,
    my_denom_ptr,
    t,
    u,
    blank_idx,
    my_stride,
    dict_size,
    ACC_DTYPE: tl.constexpr,
):
    # Log prob of emitting blank at (t, u)
    return _log_softmax_x(
        my_x_ptr, my_denom_ptr, t, u, blank_idx, my_stride, dict_size, ACC_DTYPE
    )


@triton.jit
def log_emit_x(
    my_x_ptr,
    my_denom_ptr,
    my_label_ptr,
    my_flen,
    t,
    u,
    my_stride,
    dict_size,
    dp_lam,
    eos_lam,
    eos_idx,
    dp_mu,
    dp_gamma,
    ACC_DTYPE: tl.constexpr,
):
    # Log prob of emitting the correct label at (t, u)
    # label[u] is the token emitted when moving from u -> u+1
    dp = frac_penalty(dp_lam, t, my_flen, dp_mu, dp_gamma)
    label_u = tl.load(my_label_ptr + u)
    lsmx_dp = (
        _log_softmax_x(
            my_x_ptr, my_denom_ptr, t, u, label_u, my_stride, dict_size, ACC_DTYPE
        )
        + dp
    )
    return tl.where(
        label_u == eos_idx,
        lsmx_dp + frac_penalty(eos_lam, t, my_flen, dp_mu, dp_gamma),
        lsmx_dp,
    )


@triton.jit
def transducer_loss_fwd(
    x_ptr,
    denom_ptr,
    label_ptr,
    audio_len_ptr,
    txt_len_ptr,
    batch_offset_ptr,
    dict_size: int,
    max_flen: int,
    max_glen: int,
    blank_idx: tl.constexpr,
    eos_idx: tl.constexpr,
    dp_lam: float,
    eos_lam: float,
    dp_mu,
    dp_gamma,
    alpha_ptr,
    beta_ptr,
    loss_ptr,
    U_TILE_SIZE: tl.constexpr,
    PACKED: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    alpha_or_beta = tl.program_id(0)  # 0 = alpha, 1 = beta
    batch = tl.program_id(1)

    # Batch-specific lengths
    my_flen = tl.load(audio_len_ptr + batch)
    my_glen = tl.load(txt_len_ptr + batch) + 1  # +1 for SOS token

    # x/denom use the packed offset+stride; alpha/beta always use the unpacked layout
    if PACKED:
        my_x_offset = tl.where(
            batch == 0, 0, tl.load(batch_offset_ptr + tl.maximum(batch - 1, 0))
        )
        my_stride = my_glen
    else:
        my_x_offset = batch * max_flen * max_glen
        my_stride = max_glen

    my_ab_offset = batch * max_flen * max_glen
    my_x_ptr = x_ptr + my_x_offset.to(tl.int64) * dict_size
    my_denom_ptr = denom_ptr + my_x_offset
    my_label_ptr = label_ptr + batch * (max_glen - 1)
    my_alpha_ptr = alpha_ptr + my_ab_offset
    my_beta_ptr = beta_ptr + my_ab_offset

    if alpha_or_beta == 0:
        # Initialize alpha[0, 0] = 0.0 (log-prob of empty path)
        tl.store(my_alpha_ptr, 0.0)
        tl.debug_barrier()

        for step in tl.range(1, my_flen + my_glen - 1):
            for u_start in tl.range(0, max_glen, U_TILE_SIZE):
                u_vec = u_start + tl.arange(0, U_TILE_SIZE)
                t_vec = step - u_vec
                valid = (u_vec < my_glen) & (t_vec >= 0) & (t_vec < my_flen)

                # Clamp indices to avoid OOB pointer arithmetic
                # (invalid cells are masked in the store, values are discarded)
                t_c = tl.maximum(tl.minimum(t_vec, my_flen - 1), 0)
                u_c = tl.maximum(tl.minimum(u_vec, my_glen - 1), 0)
                t_prev = tl.maximum(t_c - 1, 0)
                u_prev = tl.maximum(u_c - 1, 0)

                # Blank transition: alpha[t-1, u] + log_null_x(t-1, u)
                lnx = log_null_x(
                    my_x_ptr,
                    my_denom_ptr,
                    t_prev,
                    u_c,
                    blank_idx,
                    my_stride,
                    dict_size,
                    ACC_DTYPE,
                )
                from_blank = lnx + tl.load(
                    my_alpha_ptr + t_prev * max_glen + u_c,
                    mask=valid & (t_vec > 0),
                    other=float("-inf"),
                )

                # Label emission: alpha[t, u-1] + log_emit_x(t, u-1)
                lex = log_emit_x(
                    my_x_ptr,
                    my_denom_ptr,
                    my_label_ptr,
                    my_flen,
                    t_c,
                    u_prev,
                    my_stride,
                    dict_size,
                    dp_lam,
                    eos_lam,
                    eos_idx,
                    dp_mu,
                    dp_gamma,
                    ACC_DTYPE,
                )
                from_emit = lex + tl.load(
                    my_alpha_ptr + t_c * max_glen + u_prev,
                    mask=valid & (u_vec > 0),
                    other=float("-inf"),
                )

                new_alpha = logaddexp(from_blank, from_emit)
                tl.store(my_alpha_ptr + t_c * max_glen + u_c, new_alpha, mask=valid)

            tl.debug_barrier()

    else:
        # Initialize beta[T-1, U-1] = log_null_x(T-1, U-1)
        beta_init = log_null_x(
            my_x_ptr,
            my_denom_ptr,
            my_flen - 1,
            my_glen - 1,
            blank_idx,
            my_stride,
            dict_size,
            ACC_DTYPE,
        )
        tl.store(my_beta_ptr + (my_flen - 1) * max_glen + (my_glen - 1), beta_init)
        tl.debug_barrier()

        for step in tl.range(my_flen + my_glen - 3, -1, -1):
            for u_start in tl.range(0, max_glen, U_TILE_SIZE):
                u_vec = u_start + tl.arange(0, U_TILE_SIZE)
                t_vec = step - u_vec
                valid = (u_vec < my_glen) & (t_vec >= 0) & (t_vec < my_flen)

                # Clamp indices to avoid OOB pointer arithmetic
                t_c = tl.maximum(tl.minimum(t_vec, my_flen - 1), 0)
                u_c = tl.maximum(tl.minimum(u_vec, my_glen - 1), 0)
                t_next = tl.minimum(t_c + 1, max_flen - 1)
                u_next = tl.minimum(u_c + 1, max_glen - 1)
                # Clamp for label lookup: label has max_glen-1 entries.
                u_c_emit = tl.minimum(u_c, max_glen - 2)

                # Blank transition: beta[t+1, u] + log_null_x(t, u)
                lnx = log_null_x(
                    my_x_ptr,
                    my_denom_ptr,
                    t_c,
                    u_c,
                    blank_idx,
                    my_stride,
                    dict_size,
                    ACC_DTYPE,
                )
                from_blank = lnx + tl.load(
                    my_beta_ptr + t_next * max_glen + u_c,
                    mask=valid & (t_vec < my_flen - 1),
                    other=float("-inf"),
                )

                # Label emission: beta[t, u+1] + log_emit_x(t, u)
                lex = log_emit_x(
                    my_x_ptr,
                    my_denom_ptr,
                    my_label_ptr,
                    my_flen,
                    t_c,
                    u_c_emit,
                    my_stride,
                    dict_size,
                    dp_lam,
                    eos_lam,
                    eos_idx,
                    dp_mu,
                    dp_gamma,
                    ACC_DTYPE,
                )
                from_emit = lex + tl.load(
                    my_beta_ptr + t_c * max_glen + u_next,
                    mask=valid & (u_vec < my_glen - 1),
                    other=float("-inf"),
                )

                new_beta = logaddexp(from_blank, from_emit)
                tl.store(my_beta_ptr + t_c * max_glen + u_c, new_beta, mask=valid)

            tl.debug_barrier()

        # loss[batch] = -beta[0, 0]
        tl.store(loss_ptr + batch, -tl.load(my_beta_ptr))


@triton.autotune(
    configs=[
        triton.Config({"H_TILE_SIZE": 32}, num_warps=1),
        triton.Config({"H_TILE_SIZE": 64}, num_warps=2),
        triton.Config({"H_TILE_SIZE": 128}, num_warps=4),
        triton.Config({"H_TILE_SIZE": 256}, num_warps=8),
        triton.Config({"H_TILE_SIZE": 512}, num_warps=16),
        triton.Config({"H_TILE_SIZE": 1024}, num_warps=32),
    ],
    key=["dict_size"],
)
@triton.jit
def transducer_loss_bwd(
    x_ptr,
    denom_ptr,
    loss_grad_ptr,
    label_ptr,
    audio_len_ptr,
    txt_len_ptr,
    batch_offset_ptr,
    dict_size: int,
    max_flen: int,
    max_glen: int,
    blank_idx: tl.constexpr,
    eos_idx: tl.constexpr,
    dp_lam: float,
    eos_lam: float,
    dp_mu,
    dp_gamma,
    alpha_ptr,
    beta_ptr,
    x_grad_ptr,
    H_TILE_SIZE: tl.constexpr,
    PACKED: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    # Grid: (max_glen, max_flen, B) — one program per (u, t, batch) cell
    u = tl.program_id(0)
    t = tl.program_id(1)
    batch = tl.program_id(2)

    my_flen = tl.load(audio_len_ptr + batch)
    my_glen = tl.load(txt_len_ptr + batch) + 1  # +1 for SOS token

    # x/denom/x_grad use the packed offset+stride; alpha/beta always use the unpacked layout
    my_x_offset = (
        tl.where(batch == 0, 0, tl.load(batch_offset_ptr + tl.maximum(batch - 1, 0)))
        if PACKED
        else batch * max_flen * max_glen
    )
    my_ab_offset = batch * max_flen * max_glen
    my_stride = my_glen if PACKED else max_glen
    my_tu_offset = my_x_offset + t * my_stride + u
    my_x_ptr = x_ptr + my_tu_offset.to(tl.int64) * dict_size
    my_label_ptr = label_ptr + batch * (max_glen - 1)
    my_alpha_ptr = alpha_ptr + my_ab_offset
    my_beta_ptr = beta_ptr + my_ab_offset
    my_x_grad_ptr = x_grad_ptr + my_tu_offset.to(tl.int64) * dict_size

    if t < my_flen and u < my_glen:
        # In packed mode my_stride = my_glen (actual per-batch U+1), so threads with
        # t >= my_flen would compute my_tu_offset past the end of the packed buffer.
        # Defer the denom load until we know (t, u) is in-bounds.
        my_denom = tl.load(denom_ptr + my_tu_offset)
        # Scalar values shared across the h loop
        log_P = tl.load(my_beta_ptr)  # beta[0, 0] = log P(y|x)
        common = (
            tl.log(tl.load(loss_grad_ptr + batch))
            + tl.load(my_alpha_ptr + t * max_glen + u)
            - log_P
        )
        beta_TU = tl.load(my_beta_ptr + t * max_glen + u)

        is_last_u = u >= my_glen - 1
        is_last_t = t >= my_flen - 1
        is_terminal = is_last_u & is_last_t

        # Safe neighbour indices to avoid OOB loads
        u_next = tl.minimum(u + 1, max_glen - 1)
        t_next = tl.minimum(t + 1, max_flen - 1)
        u_label_safe = tl.minimum(u, max_glen - 2)  # label[u], clamped

        # beta_TUp1 = my_beta[t*max_glen+u+1] + frac_penalty(dp_lam, t, my_flen)
        beta_TUp1 = tl.load(my_beta_ptr + t * max_glen + u_next) + frac_penalty(
            dp_lam, t, my_flen, dp_mu, dp_gamma
        )
        beta_Tp1U = tl.load(my_beta_ptr + t_next * max_glen + u)
        my_Up1_label = tl.load(my_label_ptr + u_label_safe)
        # Additional eos penalty on the emit edge when label[u] == eos_idx
        if my_Up1_label == eos_idx:
            beta_TUp1 += frac_penalty(eos_lam, t, my_flen, dp_mu, dp_gamma)

        for h_start in tl.range(0, dict_size, H_TILE_SIZE):
            h_vec = h_start + tl.arange(0, H_TILE_SIZE)
            h_mask = h_vec < dict_size

            # Cast x to acc dtype to match denom/alpha/beta
            x_val = tl.load(my_x_ptr + h_vec, mask=h_mask, other=0.0).to(ACC_DTYPE)
            grad = common + sub_or_nan(x_val, my_denom)

            # Positive term: exp(grad + beta[t, u])
            my_grad = tl.exp(grad + beta_TU)

            # Negative emit term: subtract exp(grad + beta[t, u+1]) at h == label[u]
            emit_contrib = tl.where(
                ~is_last_u & (h_vec == my_Up1_label),
                tl.exp(grad + beta_TUp1),
                0.0,
            )
            my_grad -= emit_contrib

            # Negative blank term: subtract at h == blank_idx
            blank_contrib = tl.where(
                is_terminal,
                tl.exp(grad),  # terminal cell: no beta_Tp1U
                tl.where(~is_last_t, tl.exp(grad + beta_Tp1U), 0.0),
            )
            my_grad -= tl.where(h_vec == blank_idx, blank_contrib, 0.0)

            tl.store(my_x_grad_ptr + h_vec, my_grad, mask=h_mask)
    elif not PACKED:
        # Zero gradients for don't-care cells
        for h_start in tl.range(0, dict_size, H_TILE_SIZE):
            h_vec = h_start + tl.arange(0, H_TILE_SIZE)
            tl.store(
                my_x_grad_ptr + h_vec,
                tl.zeros((H_TILE_SIZE,), dtype=tl.float32),
                mask=h_vec < dict_size,
            )


@beartype
class TritonTransducerLossFunc(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x,
        label,
        f_len,
        y_len,
        batch_offset,
        max_f_len,
        blank_idx,
        penalty,
        packed_input,
    ):
        dict_size = x.size(-1)
        B = label.size(0)
        max_glen = label.size(1) + 1
        max_flen = max_f_len if packed_input else x.size(1)

        if packed_input:
            denom = logsumexp_cu.logsumexp(x, 128, True)
        else:
            denom = logsumexp_cu.logsumexp(x.view(-1, dict_size), 128, True).view(
                B, max_flen, max_glen
            )

        # Acc dtype: fp32 for fp16/bf16/fp32 inputs, fp64 for fp64 (mirrors at::acc_type)
        acc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        acc_triton_dtype = _TORCH_TO_TRITON_DTYPE[acc_dtype]
        alpha = torch.empty(B, max_flen, max_glen, device=x.device, dtype=acc_dtype)
        beta = torch.empty(B, max_flen, max_glen, device=x.device, dtype=acc_dtype)
        loss = torch.empty(B, device=x.device, dtype=acc_dtype)

        U_TILE_SIZE = triton.next_power_of_2(min(1024, max_glen))
        grid = (2, B)
        transducer_loss_fwd[grid](
            x,
            denom,
            label,
            f_len,
            y_len,
            batch_offset,
            dict_size,
            max_flen,
            max_glen,
            blank_idx,
            penalty.eos_idx,
            penalty.dp_lam,
            penalty.eos_lam,
            penalty.dp_mu,
            penalty.dp_gamma,
            alpha,
            beta,
            loss,
            U_TILE_SIZE,
            packed_input,
            acc_triton_dtype,
        )

        ctx.save_for_backward(x, denom, alpha, beta, label, f_len, y_len, batch_offset)
        ctx.blank_idx = blank_idx
        ctx.penalty = penalty
        ctx.packed_input = packed_input

        return loss, alpha, beta

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, loss_grad, _alpha_grad, _beta_grad):
        x, denom, alpha, beta, label, f_len, y_len, batch_offset = ctx.saved_tensors
        blank_idx = ctx.blank_idx
        penalty = ctx.penalty
        packed_input = ctx.packed_input
        B, max_flen, max_glen = alpha.shape
        dict_size = x.size(-1)

        acc_triton_dtype = _TORCH_TO_TRITON_DTYPE[alpha.dtype]
        x_grad = torch.empty_like(x)

        grid = (max_glen, max_flen, B)
        # .contiguous() guards against stride-0 broadcast tensors from .sum().backward()
        transducer_loss_bwd[grid](
            x,
            denom,
            loss_grad.contiguous(),
            label,
            f_len,
            y_len,
            batch_offset,
            dict_size,
            max_flen,
            max_glen,
            blank_idx,
            penalty.eos_idx,
            penalty.dp_lam,
            penalty.eos_lam,
            penalty.dp_mu,
            penalty.dp_gamma,
            alpha,
            beta,
            x_grad,
            PACKED=packed_input,
            ACC_DTYPE=acc_triton_dtype,
        )

        return x_grad, None, None, None, None, None, None, None, None


@beartype
class TritonTransducerLoss(torch.nn.Module):
    """
    A Triton implementation of the RNN-T loss.
    """

    def __init__(
        self,
        packed_input: bool = False,
        dp_mu: float = 0.0,
        dp_clamp: Optional[float] = None,
    ):
        super().__init__()
        self.packed_input = packed_input
        self.dp_mu = dp_mu
        self.dp_clamp = dp_clamp

    @beartype
    def forward(
        self,
        x: torch.Tensor,
        label: torch.Tensor,
        f_len: torch.Tensor,
        y_len: torch.Tensor,
        blank_idx: int,
        eos_idx: Optional[int] = None,
        star_idx: Optional[int] = None,
        batch_offset: Optional[torch.Tensor] = None,
        max_f_len: Optional[int] = None,
        debug_list: Optional[List[torch.Tensor]] = None,
        delay_penalty: float = 0.0,
        eos_penalty: float = 0.0,
        star_penalty: float = 1.0,
    ) -> torch.Tensor:
        assert len(x.shape) == 4 or len(x.shape) == 2, "Shape (B, T, U, H) or (*, H)"
        assert star_idx is None, "star penalty is not supported"
        assert f_len.min() >= 1, "f_len must be non-negative"
        assert y_len.min() >= 0, "y_len must be non-negative"
        assert y_len.max() <= label.size(1), "y_len must be less than label length"

        if self.packed_input:
            if batch_offset is None or max_f_len is None:
                raise Exception(
                    "Please specify batch_offset and max_f_len when packing is enabled"
                )
            my_batch_offset = batch_offset
            my_max_f_len = max_f_len
            assert my_max_f_len == f_len.max()
        else:
            # Dummy tensor on the correct device; never loaded from in non-packed kernels
            my_batch_offset = torch.empty(0, dtype=torch.int64, device=x.device)
            my_max_f_len = x.size(1)

        penalty = PenaltyParams(
            dp_lam=delay_penalty,
            eos_lam=eos_penalty,
            eos_idx=-1 if eos_idx is None else eos_idx,  # sentinel -1 matches CUDA
            dp_mu=self.dp_mu,
            dp_gamma=self.dp_clamp if self.dp_clamp is not None else NO_CLAMP,
        )

        loss, alpha, beta = TritonTransducerLossFunc.apply(
            x,
            label,
            f_len,
            y_len,
            my_batch_offset,
            my_max_f_len,
            blank_idx,
            penalty,
            self.packed_input,
        )

        if debug_list == []:
            debug_list += [alpha, beta]

        return loss
