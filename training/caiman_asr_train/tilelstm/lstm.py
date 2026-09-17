import math
from typing import Optional

import torch
from beartype import beartype
from torch import Tensor, nn

from caiman_asr_train.autotune.dtypes import accum_dtype_for
from caiman_asr_train.tilelstm.autotune import (
    select_bwd_train_kernel,
    select_fwd_kernel,
    select_fwd_train_kernel,
)

# -------------------------------------------------------------------------------
#
# TileLangLSTM: a drop-in LSTM whose recurrent scan runs a tuned TileLang kernel.
#
# -------------------------------------------------------------------------------


@beartype
def _accum_dtype(x: Tensor) -> torch.dtype:
    return accum_dtype_for(x.dtype)


class _TileLangRecurrentFn(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    @beartype
    def forward(
        ctx,
        gates: Tensor,
        R: Tensor,
        Y0: Tensor,
        C0: Tensor,
        fwd_kernel,
        bwd_kernel,
        Ys: Tensor,
        Cs: Tensor,
    ) -> Tensor:
        # Writes Ys (lo) and Cs (fp16) in place; Ys[0]/Cs[0] are the seeds, the outputs
        # land in slots 1..S. When training (bwd_kernel given) the gate-saving forward
        # also saves the post-activation gates the backward reads -- written back INTO
        # the ``gates`` projection buffer IN PLACE (``As`` aliases ``Gs``), so there is
        # no separate activation buffer. Returns the hidden states for t = 1..S.
        fwd_kernel(Y0, C0, R, gates, Ys, Cs)
        if bwd_kernel is not None:
            # gates now holds the post-activation gates; save it (as As) + the cell
            # states the bwd kernel reads, the recurrent weight, and Ys for dR.
            ctx.save_for_backward(gates, Cs, R, Ys)
            ctx.bwd_kernel = bwd_kernel

        return Ys[1:]

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.amp.custom_bwd(device_type="cuda")
    @beartype
    def backward(ctx, dy: Tensor) -> tuple:
        # dy = grad of Ys[1:], i.e. delta [S, B, H].
        As, Cs, R, Ys = ctx.saved_tensors
        S, B, H = dy.shape

        # dGates carries an extra (S+1)-th zero slot the kernel reads as the first
        # reversed step's recurrent term (so every step is uniform); see lstm_bwd.
        # The kernel OVERWRITES slots 0..S-1, so only the pad slot S needs zeroing
        dGates = torch.empty(S + 1, B, 4, H, device=dy.device, dtype=As.dtype)
        dGates[S].zero_()
        delta = dy.to(As.dtype, memory_format=torch.contiguous_format)
        ctx.bwd_kernel(As, Cs, R, delta, dGates)
        dgates = dGates[:S]  # grad of the gates input [S, B, 4, H]

        # dR[k, n, m] = sum_{t,b} dGates[t,b,k,n] * Ys[t,b,m], with Ys[:-1] the
        # hidden states h[0..S-1] that fed each step's recurrent gemm. Since
        # dR_flat[k*H+n, m] = sum_{t,b} dgates[t,b,k*H+n] * Ys[t,b,m], this is just
        # dgates^T @ Ys reshaped to [4,H,H].
        dR = torch.mm(
            dgates.view(S * B, 4 * H).t(),
            Ys[:-1].view(S * B, H),
            out_dtype=_accum_dtype(dgates),
        )

        # Align with apply's args: gates, R, Y, C0, fwd_kernel, bwd_kernel, Ys, Cs.
        # Initial-state grads (Y, C0) are not propagated.
        return dgates, dR.view(4, H, H), None, None, None, None, None, None


class _TileLangLSTMLayer(nn.Module):
    @beartype
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        kwargs = {"dtype": dtype, "device": device, "requires_grad": True}

        self.weight_ih = nn.Parameter(
            torch.empty(4 * hidden_size, input_size, **kwargs)
        )
        self.weight_hh = nn.Parameter(
            torch.empty(4 * hidden_size, hidden_size, **kwargs)
        )
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size, **kwargs))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size, **kwargs))

        rsh = 1.0 / math.sqrt(hidden_size)
        with torch.no_grad():
            for p in self.parameters():
                p.uniform_(-rsh, rsh)

    @beartype
    def project(self, x: Tensor) -> Tensor:
        """
        Input projection ``W_ih @ x_t + (b_ih + b_hh)`` for every timestep, in
        one matmul -- the parallel half of the layer, kept separate from the
        recurrent ``scan`` so the module can size its state buffers from the real
        projection dtype before running any layer.

        Returns the gate-major gates ``[S, B, 4, H]``. The kernel adds the recurrent
        contribution and applies the nonlinearities.
        """
        S, B, _ = x.shape
        H = self.hidden_size
        x_flat = x.reshape(S * B, self.input_size)
        gates = torch.addmm(self.bias_ih + self.bias_hh, x_flat, self.weight_ih.t())
        return gates.view(S, B, 4, H).contiguous()

    @beartype
    def scan(
        self,
        gates: Tensor,
        h0: Tensor,
        c0: Tensor,
        Ys: Tensor,
        Cs: Tensor,
        train: bool,
    ) -> Tensor:
        """
        Run the recurrent scan over ``gates``, writing the hidden/cell states into
        the caller-provided buffers ``Ys``/``Cs`` in place. Returns the hidden states
        for t = 1..S (the kernel's ``Ys[1:]``).

        When ``train`` the gate-saving forward also writes the post-activation gates
        the backward reads -- back INTO ``gates`` in place (no separate ``As`` buffer;
        see ``_TileLangRecurrentFn.forward``). ``Ys``/``Cs`` carry the seed in slot 0
        (the kernel writes it from ``Y0``/``C0``), so the full ``[S+1]`` slices pass.

        The recurrent weight and hidden seed are cast to the operand dtype ``lo``
        (a no-op outside autocast); the cell seed is cast to the compute dtype the
        kernel accumulates the cell in -- ``fp32`` for any fp16/bf16/fp32 input,
        promoted to ``fp64`` only when the seed is itself fp64 (see ``_accum_dtype``).
        """
        _, B, _, H = gates.shape
        device = gates.device
        lo_dtype = gates.dtype
        # Compute/accumulator dtype, decoupled from Cs's fp16 storage (see above):
        # fp32 for fp16/bf16/fp32 seeds, fp64 only when the cell seed is itself fp64.
        hi_dtype = _accum_dtype(c0)

        assert (
            gates.dtype == Ys.dtype
        ), f"gate dtype {gates.dtype} != hidden-buffer dtype {Ys.dtype}"

        R = self.weight_hh.to(lo_dtype, memory_format=torch.contiguous_format).view(
            4, H, H
        )
        Y0 = h0.to(lo_dtype, memory_format=torch.contiguous_format)
        C0 = c0.to(hi_dtype, memory_format=torch.contiguous_format)

        if train:
            # Training: the gate-saving forward overwrites ``gates`` with the
            # post-activation gates (reused as ``As``), and the tuned backward kernel
            # is threaded through for _TileLangRecurrentFn.backward.
            fwd_kernel = select_fwd_train_kernel(H, B, device, lo_dtype, hi_dtype)
            bwd_kernel = select_bwd_train_kernel(H, B, device, lo_dtype, hi_dtype)
            return _TileLangRecurrentFn.apply(
                gates, R, Y0, C0, fwd_kernel, bwd_kernel, Ys, Cs
            )

        fwd_kernel = select_fwd_kernel(H, B, device, lo_dtype, hi_dtype)
        return _TileLangRecurrentFn.apply(gates, R, Y0, C0, fwd_kernel, None, Ys, Cs)

    @beartype
    def forward(self, x: Tensor, state: tuple[Tensor, Tensor]) -> Tensor:
        """
        Standalone single-layer forward: project, allocate this layer's own scan
        buffers, and run. The multilayer ``TileLangLSTM`` does NOT use this path -- it
        calls ``project``/``scan`` directly with slices of one shared allocation (no
        per-layer alloc) -- but it keeps the layer a well-formed, independently callable
        ``nn.Module`` (used by the unit tests).
        """
        h0, c0 = state
        S, B, _ = x.shape
        H = self.hidden_size
        gates = self.project(x)
        lo_dtype = gates.dtype
        Ys = torch.empty(S + 1, B, H, device=x.device, dtype=lo_dtype)
        # Cell buffer stored fp16 (lo); the kernel keeps the running cell in fp32 inside.
        Cs = torch.empty(S + 1, B, H, device=x.device, dtype=lo_dtype)
        return self.scan(gates, h0, c0, Ys, Cs, torch.is_grad_enabled())


class TileLangLSTM(nn.Module):
    """
    Drop-in LSTM matching ``CustomLSTM``'s factory args and return signature.

    Returns the same 3-tuple as ``CustomLSTM.forward``:
    ``(output, (h_n, c_n), (all_h, all_c))`` where ``output`` is the final layer's
    hidden sequence ``[S, B, H]``, ``h_n``/``c_n`` are the per-layer final
    hidden/cell states ``[L, B, H]``, and ``all_h``/``all_c`` are every per-layer
    hidden/cell state ``[L, S, B, H]``.
    """

    @beartype
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        hard: bool = False,
        quantize: bool = False,
        rw_dropout: float = 0.0,
        dtype=None,
        device=None,
    ):
        super().__init__()
        assert not hard, "TileLangLSTM only supports soft activations (hard=False)"
        assert not quantize, "TileLangLSTM does not support quantization"
        assert rw_dropout == 0.0, "TileLangLSTM does not support recurrent weight drop"

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self._drop = (
            nn.Dropout(p=dropout)
            if (dropout > 0.0 and num_layers > 1)
            else nn.Identity()
        )

        layers = []
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            layers.append(
                _TileLangLSTMLayer(in_dim, hidden_size, dtype=dtype, device=device)
            )
        # A plain list: the layers must NOT be registered as
        self.layers = layers

        # Flat PyTorch-LSTM parameter names -> CustomLSTM-compatible state_dict.
        for i, layer in enumerate(self.layers):
            self.register_parameter(f"weight_ih_l{i}", layer.weight_ih)
            self.register_parameter(f"weight_hh_l{i}", layer.weight_hh)
            self.register_parameter(f"bias_ih_l{i}", layer.bias_ih)
            self.register_parameter(f"bias_hh_l{i}", layer.bias_hh)

    @beartype
    def forward(
        self, input: Tensor, state: Optional[tuple[Tensor, Tensor]] = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        S, B, _ = input.shape
        H = self.hidden_size
        L = self.num_layers

        if state is None:
            h0 = input.new_zeros(L, B, H)
            c0 = input.new_zeros(L, B, H)
        else:
            h0, c0 = state

        # Project layer 0 first so the operand (``lo``) dtype follows the real addmm
        # output.
        gates0 = self.layers[0].project(input)
        lo_dtype = gates0.dtype
        hi_dtype = _accum_dtype(c0)

        # Select (tuning on first use) the kernels BEFORE allocating the per-layer
        # state buffers below. The selectors are @cache'd on (H, B, device, dtypes) --
        # which every layer shares -- so each layer's scan() hits the cache. Warming
        # them here means the autotuner's (memory-heavy) validate runs with the full
        # device free, instead of with all layers' Ys/Cs already resident.
        train = torch.is_grad_enabled()
        if train:
            select_fwd_train_kernel(H, B, input.device, lo_dtype, hi_dtype)
            select_bwd_train_kernel(H, B, input.device, lo_dtype, hi_dtype)
        else:
            select_fwd_kernel(H, B, input.device, lo_dtype, hi_dtype)

        Ys_all = torch.empty(L, S + 1, B, H, device=input.device, dtype=lo_dtype)
        Cs_all = torch.empty(L, S + 1, B, H, device=input.device, dtype=lo_dtype)

        x = input
        for i, layer in enumerate(self.layers):
            if i > 0:
                x = self._drop(x)
            gates = gates0 if i == 0 else layer.project(x)
            # Match CustomLSTM: detach a *provided* initial state (truncated BPTT);
            # the kernel does not propagate grads to the seed regardless.
            h0i = h0[i].detach() if state is not None else h0[i]
            c0i = c0[i].detach() if state is not None else c0[i]
            x = layer.scan(gates, h0i, c0i, Ys_all[i], Cs_all[i], train)

        return x, (Ys_all[:, -1], Cs_all[:, -1]), (Ys_all[:, 1:], Cs_all[:, 1:])
