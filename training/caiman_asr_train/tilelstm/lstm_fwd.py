"""
TileLang LSTM forward recurrent-scan kernels.

Both kernels compute the same recurrence over an ``[S, M, H]`` problem; the
per-step, per-tile math is the recurrent gemm plus ``fwd_epilogue`` below:

    recur[:, g] = Ys[t] @ rW[g]^T            # [bM, 4*bN] slab, fp32 accum
    i, f, g, o  = act(recur + Gs[t])         # sigmoid/sigmoid/tanh/sigmoid
    C           = f * C_prev + i * g         # cell update, fp32
    Ys[t+1]     = o * tanh(C)                # hidden out, fp16

Time is inherently serial and cross-block dependent: step t's gemm reads the
full Ys[t] row of its m-group, written at t-1 by all n_prod = cdiv(H, block_N)
column tiles of that band. Both kernels stay on-device for the whole scan by
launching cooperatively and ordering bands with a monotonic per-m-group
counter F (``band_wait`` / ``band_arrive`` below). They differ in the grid
strategy -- i.e. where the cell state lives between steps.

Coop -- one block per output tile; the cell never leaves registers:

    grid = (cdiv(H, block_N), cdiv(M, block_M))     # every tile gets a block
    launch cooperative:

        C_frag = C0[tile]                           # cell lives in registers
        Ys[0]  = Y0                                 # seed hidden
        sync_grid()                                 # once: publish seeds +
                                                    #   force full co-residency

        for t in 0 .. S-1:                          # SERIAL time
            if t > 0: band_wait(F, by, n_prod * t)  # my m-group finished t-1
            C_frag = step(Ys[t], C_frag)            # gemm+epilogue -> Ys[t+1]
            band_arrive(F, by)                      # I finished t

Persistent -- fixed grid, cell re-streamed via a fp32 ping-pong (Cpp); used
past coop's residency wall (B >= ~512), where one-block-per-tile no longer
fits co-resident on the device:

    grid = blocks_per_sm * num_SMs                      # independent of tile count
    launch cooperative:

        for (by, bx) in my stride of tiles: seed Ys[0], Cs[0], Cpp[0]
        sync_grid()

        for t in 0 .. S-1:
            for (by, bx) in my stride of tiles:         # a block visits MANY tiles
                                                        #   per step -> no C_frag
                if t > 0: band_wait(F, by, n_prod * t)
                C_prev = Cpp[t % 2][tile]               # re-stream previous cell
                C = step(Ys[t], C_prev)                 # gemm+epilogue -> Ys[t+1]
                Cpp[(t + 1) % 2][tile] = C              # publish for step t+1
                band_arrive(F, by)

Both store the fp16 Ys/Cs histories the backward consumes; with ``save_gates``
they also overwrite Gs[t] in place with the post-activation gates (the
training variant).
"""

import math

import tilelang
import tilelang.language as T
import torch
from beartype import beartype
from tilelang.carver.arch import driver
from torch import Tensor

from caiman_asr_train.autotune.dtypes import TLDtype

# The tilelang lowering-pass options every LSTM kernel is jitted with -- the forward
# kernels here and the three backward kernels in ``lstm_bwd.py`` (which imports this).
LSTM_PASS_CONFIGS = {
    # Aggressively reuse shared memory across phases: the matmul operand
    # tiles (Ys_shared, rW_shared) are dead before the epilogue buffers
    # (recur_shared, ...) are written, so the planner can overlap them.
    tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    # Not compatible with aggressive-shared-memory-merge
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@T.macro
def band_wait(F, by, threshold, tx):
    """
    ACQUIRE wait: block until m-group ``by`` has accrued ``threshold`` arrivals
    (``F[by, 0] >= threshold``, ``threshold == n_prod * t``), so every column tile of
    the band has published its slot for this step; then a block barrier so all threads
    observe those writes. ``F`` counters are monotonic per launch. Caller guards with
    the step (and, for persistent, the in-range) condition; thread 0 spins, the rest
    wait at the barrier.
    """
    if tx == 0:
        while T.atomic_load(F[by, 0], memory_order="acquire") < threshold:
            T.evaluate(0)
    T.sync_threads()


@T.macro
def band_arrive(F, by, tx):
    """
    RELEASE arrive: publish this tile's stores then bump the band's monotonic
    counter. ``__threadfence`` supplies the release ordering (TileLang's
    AtomicAdd degrades to relaxed below sm_90, so the fence is load-bearing);
    ``sync_threads`` orders every thread's store before the leader's add, so an
    acquire-waiting band member sees a fully-written tile.
    """
    T.call_extern("handle", "__threadfence")
    T.sync_threads()
    if tx == 0:
        T.atomic_add(F[by, 0], 1, memory_order="release")


@T.macro
def fwd_epilogue(
    recur_local,
    Gs_shared,
    ig,
    fg,
    cg,
    og,
    Cprev,
    C_frag,
    Y_frag,
    As_shared,
    save_gates,
    block_M,
    block_N,
    dtype_lo,
    dtype_hi,
):
    """
    Shared coop/persistent per-tile epilogue: gather the recurrent-gemm accumulator
    into per-gate fragments, wait for the prefetched epilogue operands (issued by
    the caller: the Gs slab, plus the ping-pong cell for persistent), then run the
    element-wise gate activations and cell update. ``Cprev`` is the previous-cell
    source -- the register-resident ``C_frag`` itself for coop, the re-streamed
    ``Cp_shared`` for persistent; ``C_frag``/``Y_frag`` receive the new cell/hidden
    tile.
    """
    recur_view = T.view(recur_local, (block_M, 4, block_N))

    # Gather the 4 gates in registers: one single-index copy pass each
    # (same lane under FullRow -> register moves, no shared transpose).
    for i, j in T.Parallel(block_M, block_N):
        ig[i, j] = recur_view[i, 0, j]

    for i, j in T.Parallel(block_M, block_N):
        fg[i, j] = recur_view[i, 1, j]

    for i, j in T.Parallel(block_M, block_N):
        cg[i, j] = recur_view[i, 2, j]

    for i, j in T.Parallel(block_M, block_N):
        og[i, j] = recur_view[i, 3, j]

    # Wait for the caller's prefetched cp.async group to complete before the
    # cell-update loop reads it. Don't need sync_threads because of FullRow.
    T.ptx_wait_group(0)

    for i, j in T.Parallel(block_M, block_N):
        # Input-projection contribution for each gate (i, f, g, o),
        # from the prefetched shared slab.
        x_i = T.cast(Gs_shared[i, 0, j], dtype_hi)
        x_f = T.cast(Gs_shared[i, 1, j], dtype_hi)
        x_g = T.cast(Gs_shared[i, 2, j], dtype_hi)
        x_o = T.cast(Gs_shared[i, 3, j], dtype_hi)

        gate_i = T.sigmoid(ig[i, j] + x_i)
        gate_f = T.sigmoid(fg[i, j] + x_f)
        gate_g = T.tanh(cg[i, j] + x_g)
        gate_o = T.sigmoid(og[i, j] + x_o)

        if save_gates:
            # Save the post-activation gates (gate-major) for the backward.
            As_shared[i, 0, j] = T.cast(gate_i, dtype_lo)
            As_shared[i, 1, j] = T.cast(gate_f, dtype_lo)
            As_shared[i, 2, j] = T.cast(gate_g, dtype_lo)
            As_shared[i, 3, j] = T.cast(gate_o, dtype_lo)

        c_new = gate_f * Cprev[i, j] + gate_i * gate_g
        C_frag[i, j] = c_new
        Y_frag[i, j] = T.cast(gate_o * T.tanh(c_new), dtype_lo)


@tilelang.jit(pass_configs=LSTM_PASS_CONFIGS)
@beartype
def lstm_coop_fwd(
    H: int,
    block_M: int = 64,
    block_N: int = 32,  # 32 fits 100KB GPUs (4090/A6000); 64 needs A100 SMEM
    block_K: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    fpad: int = 1,
    dtype_lo: TLDtype = torch.float16,
    dtype_hi: TLDtype = torch.float32,
    save_gates: bool = False,
):
    """
    Returns an in-place LSTM recurrent scan kernel.

    Arguments:
        fpad: flags-padding: padding (in multiples of i32)
        save_gates: if True, the kernel writes the post-activation gates (i, f, g, o)
    """
    assert H % block_K == 0, "H must be divisible by block_K"

    # The wide gemm produces a [block_M, 4*block_N] slab per hidden-output tile,
    # laid out gate-major: 4 contiguous block_N segments, one per gate (i, f, g, o).
    quads_N = 4 * block_N

    # Column blocks per m-group (band members), counting the ragged last tile.
    n_prod = T.cdiv(H, block_N)
    i32 = "int32"  # named: flake8 rejects string literals in TIR annotations

    M = T.dynamic("m")
    S = T.dynamic("s")

    @T.prim_func
    def main(
        Y0: T.Tensor[(M, H), dtype_lo],
        C0: T.Tensor[(M, H), dtype_hi],
        rW: T.Tensor[(4, H, H), dtype_lo],
        Gs: T.Tensor[(S, M, 4, H), dtype_lo],
        Ys: T.Tensor[(S + 1, M, H), dtype_lo],
        Cs: T.Tensor[(S + 1, M, H), dtype_lo],
        F: T.Tensor[(T.cdiv(M, block_M), fpad), i32],
    ):
        with T.Kernel(T.cdiv(H, block_N), T.cdiv(M, block_M), threads=threads) as (
            bx,
            by,
        ):
            # Thread's index, for the band-counter wait/arrive (thread 0 only).
            tx = T.get_thread_binding()

            # MMA operand tiles
            Ys_shared = T.alloc_shared((block_M, block_K), dtype_lo)
            rW_shared = T.alloc_shared((quads_N, block_K), dtype_lo)
            # Wide dtype_hi accumulator
            recur_local = T.alloc_fragment((block_M, quads_N), dtype_hi)

            Gs_shared = T.alloc_shared((block_M, 4, block_N), dtype_lo)

            ig = T.alloc_fragment((block_M, block_N), dtype_hi)
            fg = T.alloc_fragment((block_M, block_N), dtype_hi)
            cg = T.alloc_fragment((block_M, block_N), dtype_hi)
            og = T.alloc_fragment((block_M, block_N), dtype_hi)
            Y_frag = T.alloc_fragment((block_M, block_N), dtype_lo)

            C_frag = T.alloc_fragment((block_M, block_N), dtype_hi)

            if save_gates:
                As_shared = T.alloc_shared((block_M, 4, block_N), dtype_lo)

            # ===== Seed: Ys[0] = Y0, Cs[0] = C0 (initial hidden/cell state) =====

            # Seed Ys[0] directly with a parallel copy: the epilogue stores the
            # hidden state straight from a register fragment, so there is no shared
            # output tile to stage through.
            for i, j in T.Parallel(block_M, block_N):
                Ys[0, by * block_M + i, bx * block_N + j] = Y0[
                    by * block_M + i, bx * block_N + j
                ]

            # Seed the resident cell accumulator (into registers) and write Cs[0].
            T.copy(C0[by * block_M, bx * block_N], C_frag)
            T.copy(C_frag, Cs[0, by * block_M, bx * block_N])

            # One-shot grid barrier: publishes Ys[0] for step 0's gemm AND forces the
            # cooperative-launch classification the per-step band spins below need to
            # be deadlock-free (every block must be co-resident before any spins).
            T.sync_grid()

            # ===== Steps t = 0 .. S-1: every iteration is identical =====

            for t in T.serial(S):
                T.clear(recur_local)

                if t > 0:
                    band_wait(F, by, n_prod * t, tx)

                for k in T.Pipelined(T.cdiv(H, block_K), num_stages=num_stages):
                    # Inside a Pipelined loop these copies are lowered to async
                    # copies automatically
                    T.copy(Ys[t, by * block_M, k * block_K], Ys_shared)
                    xN = bx * block_N
                    kK = k * block_K
                    T.copy(rW[0, xN, kK], rW_shared[0 * block_N : 1 * block_N, :])
                    T.copy(rW[1, xN, kK], rW_shared[1 * block_N : 2 * block_N, :])
                    T.copy(rW[2, xN, kK], rW_shared[2 * block_N : 3 * block_N, :])
                    T.copy(rW[3, xN, kK], rW_shared[3 * block_N : 4 * block_N, :])
                    T.gemm(
                        Ys_shared,
                        rW_shared,
                        recur_local,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                T.async_copy(Gs[t, by * block_M, 0, bx * block_N], Gs_shared)

                # Coop keeps the cell resident in registers: C_frag is both the
                # previous-cell source and the new-cell destination.
                fwd_epilogue(
                    recur_local,
                    Gs_shared,
                    ig,
                    fg,
                    cg,
                    og,
                    C_frag,
                    C_frag,
                    Y_frag,
                    As_shared if save_gates else None,
                    save_gates,
                    block_M,
                    block_N,
                    dtype_lo,
                    dtype_hi,
                )

                if save_gates:
                    # Overwrite Gs[t] in place with the post-activation gates (As
                    # aliases Gs): Gs[t] was async-copied to Gs_shared and read in the
                    # epilogue above, so this is a safe per-step read-before-write.
                    T.copy(As_shared, Gs[t, by * block_M, 0, bx * block_N])

                # Coalesced stores into the next slots, both straight from registers.
                T.copy(C_frag, Cs[t + 1, by * block_M, bx * block_N])
                T.copy(Y_frag, Ys[t + 1, by * block_M, bx * block_N])

                band_arrive(F, by, tx)

    return main


@tilelang.jit(pass_configs=LSTM_PASS_CONFIGS)
@beartype
def lstm_persistent_fwd(
    H: int,
    block_M: int = 64,
    block_N: int = 32,
    block_K: int = 64,
    num_stages: int = 2,
    threads: int = 128,
    blocks_per_sm: int = 2,
    fpad: int = 1,
    dtype_lo: TLDtype = torch.float16,
    dtype_hi: TLDtype = torch.float32,
    save_gates: bool = False,
):
    """
    Persistent ``T.Persistent`` LSTM scan: a fixed cooperative grid of
    ``blocks_per_sm * sm_num`` blocks, each striding over the (batch x hidden)
    output tiles. Used past coop's residency wall (B >= 512).

    Arguments:
        blocks_per_sm: Grid = blocks_per_sm * SMs
        fpad: flags-padding: padding (in multiples of i32)

    """
    assert H % block_K == 0, "H must be divisible by block_K"
    quads_N = 4 * block_N  # one block_N hidden-tile = 4 gate columns of block_N

    # Column tiles per m-group (band members), counting the ragged last tile: each
    # arrives once on its m-group's counter per completed step (mirrors coop).
    n_prod = T.cdiv(H, block_N)
    i32 = "int32"  # named: flake8 rejects string literals in TIR annotations

    grid = blocks_per_sm * driver.get_num_sms()

    # TODO: gcd not required after: https://github.com/tile-ai/tilelang/pull/2455
    persist_group = math.gcd(tilelang.cdiv(H, block_N), 8)  # column-tile count (cdiv)

    M = T.dynamic("m")
    S = T.dynamic("s")

    @T.prim_func
    def main(
        Y0: T.Tensor[(M, H), dtype_lo],
        C0: T.Tensor[(M, H), dtype_hi],
        rW: T.Tensor[(4, H, H), dtype_lo],
        Gs: T.Tensor[(S, M, 4, H), dtype_lo],
        Ys: T.Tensor[(S + 1, M, H), dtype_lo],
        Cs: T.Tensor[(S + 1, M, H), dtype_lo],
        F: T.Tensor[(T.cdiv(M, block_M), fpad), i32],
        Cpp: T.Tensor[(2, M, H), dtype_hi],
    ):
        with T.Kernel(grid, threads=threads) as block_id:
            # Thread's index, for the band-counter wait/arrive (thread 0 only).
            tx = T.get_thread_binding()
            Ys_shared = T.alloc_shared((block_M, block_K), dtype_lo)
            rW_shared = T.alloc_shared((quads_N, block_K), dtype_lo)
            recur_local = T.alloc_fragment((block_M, quads_N), dtype_hi)
            # Per-tile epilogue inputs (shared); the merge overlaps them onto the dead
            # operand tiles (Gs shares Ys's offset).
            Gs_shared = T.alloc_shared((block_M, 4, block_N), dtype_lo)
            Cp_shared = T.alloc_shared((block_M, block_N), dtype_hi)
            # Register epilogue: one fragment per gate (filled by a single-index copy
            # pass from recur_local; same-lane under FullRow) + output fragments stored
            # straight to global. No recur_shared transpose.
            ig = T.alloc_fragment((block_M, block_N), dtype_hi)
            fg = T.alloc_fragment((block_M, block_N), dtype_hi)
            cg = T.alloc_fragment((block_M, block_N), dtype_hi)
            og = T.alloc_fragment((block_M, block_N), dtype_hi)
            Y_frag = T.alloc_fragment((block_M, block_N), dtype_lo)
            C_frag = T.alloc_fragment((block_M, block_N), dtype_hi)
            if save_gates:
                As_shared = T.alloc_shared((block_M, 4, block_N), dtype_lo)

            # Seed Ys[0]/Cs[0], one block_N-tile at a time.
            for by, bx in T.Persistent(
                [T.cdiv(M, block_M), T.cdiv(H, block_N)],
                grid,
                block_id,
                persist_group,
            ):
                for i, j in T.Parallel(block_M, block_N):
                    Ys[0, by * block_M + i, bx * block_N + j] = Y0[
                        by * block_M + i, bx * block_N + j
                    ]
                    # fp16 saved-history seed (down-cast from the fp32 C0)...
                    Cs[0, by * block_M + i, bx * block_N + j] = C0[
                        by * block_M + i, bx * block_N + j
                    ]
                    # ...and the fp32 ping-pong seed in slot 0 (step 0 reads Cpp[0%2]).
                    Cpp[0, by * block_M + i, bx * block_N + j] = C0[
                        by * block_M + i, bx * block_N + j
                    ]
            T.sync_grid()

            for t in T.serial(S):
                for by, bx in T.Persistent(
                    [T.cdiv(M, block_M), T.cdiv(H, block_N)],
                    grid,
                    block_id,
                    persist_group,
                ):
                    T.clear(recur_local)

                    # In-range guard for the band counter. When the persistent grid
                    # OVER-SUBSCRIBES the tile set (grid > #tiles -> T.Persistent's
                    # ``waves == 1``), it emits NO out-of-range break: every block runs
                    # this body once, and the surplus blocks land on an out-of-range
                    # column ``bx >= cdiv(H, block_N)``. TileLang predicates their gemm
                    # operand loads / output stores off (the ``bx*block_N < H`` bound),
                    # but NOT these raw atomics -- so without this guard every surplus
                    # block would still ``arrive`` on F, inflating the count past
                    # ``n_prod*t`` and letting the acquire-waits pass before the real
                    # producers wrote Ys[t] (a timing race -> scattered NaN). ``group =
                    # gcd(CT, 8)`` divides CT, so ``bx < n_prod`` is the exact tile-in-
                    # range test (and a no-op when ``waves >= 2``, where the break
                    # already drops out-of-range iterations). ``bx`` is uniform across
                    # the block, so the guarded ``sync_threads`` is not divergent.
                    if bx < n_prod:
                        if t > 0:
                            band_wait(F, by, n_prod * t, tx)

                    for k in T.Pipelined(T.cdiv(H, block_K), num_stages=num_stages):
                        kK = k * block_K
                        T.copy(Ys[t, by * block_M, kK], Ys_shared)
                        for g in range(4):
                            T.copy(
                                rW[g, bx * block_N, kK],
                                rW_shared[g * block_N : (g + 1) * block_N, :],
                            )
                        T.gemm(
                            Ys_shared,
                            rW_shared,
                            recur_local,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

                    # The SMEM-merge aliases Gs_shared/Cp_shared onto the now-dead
                    # operand tiles (Ys_shared/rW_shared). The gemm's ldmatrix reads of
                    # those operands and the cp.async writes below hit the same shared
                    # addresses, so without a barrier the prefetch clobbers the operands
                    # while other warps are still reading them -- a cross-warp WAR race.
                    T.sync_threads()

                    T.async_copy(Gs[t, by * block_M, 0, bx * block_N], Gs_shared)
                    T.async_copy(Cpp[t % 2, by * block_M, bx * block_N], Cp_shared)

                    # Persistent re-streams the previous cell each step: the fp32
                    # ping-pong slab prefetched into Cp_shared is the c_prev source.
                    fwd_epilogue(
                        recur_local,
                        Gs_shared,
                        ig,
                        fg,
                        cg,
                        og,
                        Cp_shared,
                        C_frag,
                        Y_frag,
                        As_shared if save_gates else None,
                        save_gates,
                        block_M,
                        block_N,
                        dtype_lo,
                        dtype_hi,
                    )

                    if save_gates:
                        T.copy(As_shared, Gs[t, by * block_M, 0, bx * block_N])

                    # Save the cell history fp16 (down-cast) AND publish the fp32 cell
                    # into the ping-pong's OTHER slot for the next step's re-read.
                    T.copy(C_frag, Cs[t + 1, by * block_M, bx * block_N])
                    T.copy(C_frag, Cpp[(t + 1) % 2, by * block_M, bx * block_N])
                    T.copy(Y_frag, Ys[t + 1, by * block_M, bx * block_N])

                    if bx < n_prod:
                        band_arrive(F, by, tx)

    return main


@beartype
def lstm_ref(Y: Tensor, C0: Tensor, R: Tensor, gates: Tensor):
    """
    Pure-PyTorch transcription of ``lstm_coop_fwd``'s recurrent scan.

    Takes the *same* inputs the kernel receives -- ``R`` ``[4, H, H]`` and
    ``gates`` ``[S, B, 4, H]``, both gate-major (gate k, unit j is ``R[k, j, :]`` /
    ``gates[t, :, k, j]``, k = 0..3 -> i, f, g, o) -- and reproduces the kernel's
    precision *from the input dtypes*: the hidden state is ``lo = Y.dtype``; the
    recurrent gemm accumulates into ``hi = C0.dtype`` (matching the kernel's
    ``dtype_hi`` accumulator fragment -- on CUDA this uses ``torch.mm(out_dtype=hi)``
    for fp16-in/fp32-accum via tensor cores; on CPU inputs are up-cast to ``hi``
    first). The cell state and gate math also run in ``hi``. The list slots are the
    kernel's ``Ys`` / ``Cs`` buffers: slot 0 is the (Y, C0) seed and step t reads
    slot t, writes slot t+1. Each step is the kernel's per-step body, line for line:

        pre[:, k] = mm(Ys[t], R[k].T, out_dtype=hi) + gates[t, :, k]   # in hi
        gate_i, gate_f, gate_g, gate_o = sigmoid/sigmoid/tanh/sigmoid of pre
        Cs[t+1]   = gate_f * Cs[t] + gate_i * gate_g  # hi cell update
        Ys[t+1]   = gate_o * tanh(Cs[t+1])            # cast back to lo

    Returns (Ys lo, Cs hi), each [S + 1, B, H].
    """
    S = gates.shape[0]
    lo, hi = Y.dtype, C0.dtype  # kernel's dtype_lo / dtype_hi
    Ys = [Y.to(lo)]  # Ys[0] = Y  (hidden seed, lo)
    Cs = [C0.to(hi)]  # Cs[0] = C0 (cell seed, hi)
    for t in range(S):
        # Recurrent matmul per gate.  The kernel accumulates into a dtype_hi
        # (fp32) fragment; we match that with torch.mm(out_dtype=hi) on CUDA
        # (fp16 inputs, fp32 accumulation via tensor cores) and an explicit
        # cast to hi on CPU (where the out_dtype overload is not available).
        if Ys[t].is_cuda and Ys[t].dtype != hi:
            pre = torch.stack(
                [torch.mm(Ys[t], R[k].T, out_dtype=hi) for k in range(4)], dim=1
            )
        else:
            pre = torch.stack([Ys[t].to(hi) @ R[k].T.to(hi) for k in range(4)], dim=1)
        pre = pre + gates[t].to(hi)
        gate_i = torch.sigmoid(pre[:, 0])
        gate_f = torch.sigmoid(pre[:, 1])
        gate_g = torch.tanh(pre[:, 2])
        gate_o = torch.sigmoid(pre[:, 3])
        c_new = gate_f * Cs[t] + gate_i * gate_g
        Cs.append(c_new)
        Ys.append((gate_o * torch.tanh(c_new)).to(lo))
    return torch.stack(Ys, dim=0), torch.stack(Cs, dim=0)
