from collections.abc import Callable, Iterator
from functools import partial
from itertools import chain

import tilelang
import torch
from beartype import beartype
from tilelang.language import GemmWarpPolicy

from caiman_asr_train.autotune.device import DeviceLimits, device_limits
from caiman_asr_train.autotune.feasibility import (
    ATOM_K,
    ATOM_M,
    ATOM_N,
    TPW,
    grid_resident,
    mnk_atom_ok,
    regs_per_block,
    smem_fits,
    threads_are_warp,
    wmma_ok,
)
from caiman_asr_train.tilelstm.autotune.config_candidates import (
    bwd_candidate_tiles,
    fwd_candidate_tiles,
    splitk_candidate_tiles,
)
from caiman_asr_train.tilelstm.autotune.configs_smem import (
    coop_bwd_smem_bytes,
    coop_fwd_smem_bytes,
    persistent_bwd_smem_bytes,
    persistent_fwd_smem_bytes,
    splitk_smem_bytes_at,
)

# Padded-counter width for the flags builds: spaces the per-band / per-column
# release/acquire counters out so adjacent counters never share an L2 sector
# (false-sharing the monotonic atomics would serialise drifting bands). 32 int32
# = 128 B = one sector
_FLAGS_FPAD = 32

# Per-thread register pressure drives cooperative-launch occupancy: both kernels are
# register-heavy (wide fp16 gemm fragment + fp32 accumulator + sigmoid/tanh temporaries),
# so on a register-bound card threads=256 gets ~1 block/SM and threads=128 ~3. The
# residency estimate (``_regs_per_block``) needs a register term or it is far too
# optimistic (SMEM/threads only) and lets cooperative-launch-too-large configs through.
#
# ``_REG_FLOOR_PER_THREAD`` is the OPTIMISTIC floor in ``_regs_per_block`` (raised by the
# accumulator term for big tiles). Wrongly EXCLUDING a runnable config loses a potential
# winner forever, so the pre-prune must never do that: the floor is low enough that every
# config that actually launches survives.
_REG_FLOOR_PER_THREAD = 56


@beartype
def _tile_ok(cfg: dict) -> bool:
    """
    The non-atom shape constraints every non-split-K generator shares:

        block_N <= H        -- a column tile no wider than H (ragged, not oversized);
        block_K divides H   -- a K-chunk must lie wholly within one gate;
    """
    return cfg["block_N"] <= cfg["H"] and cfg["H"] % cfg["block_K"] == 0


@beartype
def tile_hidden_feasible(H: int) -> bool:
    """
    Whether hidden size ``H`` admits ANY TileLang LSTM tile.
    """
    atom = {"block_M": ATOM_M, "block_N": ATOM_N, "block_K": ATOM_K, "H": H}

    return mnk_atom_ok(atom, n_block_mult=1) and _tile_ok(atom)


@beartype
def _epilogue_reduction_ok(cfg: dict) -> bool:
    """
    Forward-only gate-major epilogue legality -- TWO constraints:

    ROWS: the FullRow gate epilogue reduces each gate's 4 columns within a warp, so
    every warp needs a whole >=16-row group i.e.:

      num_mma_in_M >= num_warps       (such that each warp has work)
      block_M / 16 >= threads // 32
       block_M * 2 >= threads

    Below that an MMA atom straddles two warps and the per-gate fragments
    (ig/fg/cg/og), gathered from different accumulator columns, get row->thread
    maps off by a warp-group offset; the combine loop is then unsatisfiable and
    TileLang aborts layout inference.

    COLUMNS: the 4 gates live at output columns ``g*block_N + j`` (g=0..3); the
    same-lane gather that makes the gate combine a register move needs
    ``(g*block_N + j) % 8 == j % 8`` for all g, i.e. ``block_N % 8 == 0``.
    """
    return cfg["threads"] <= 2 * cfg["block_M"] and cfg["block_N"] % 8 == 0


@beartype
def _regs_per_block(cfg: dict, n_block_mult: int) -> int:
    """
    Per-block register estimate for the residency check, with the LSTM kernels'
    calibrated per-thread register floor bound in.
    """
    return regs_per_block(
        cfg, n_block_mult=n_block_mult, reg_floor_per_thread=_REG_FLOOR_PER_THREAD
    )


@beartype
def _coop_grid_resident(
    cfg: dict, *, smem: int, lim: DeviceLimits, B: int, n_block_mult: int
) -> bool:
    """
    Whether the COOPERATIVE problem-sized grid ``cdiv(H, block_N) * cdiv(B, block_M)``
    co-resides at this SMEM/threads/registers.
    """
    grid = tilelang.cdiv(cfg["H"], cfg["block_N"]) * tilelang.cdiv(B, cfg["block_M"])

    return grid_resident(
        grid=grid,
        threads=cfg["threads"],
        smem=smem,
        lim=lim,
        regs_per_block=_regs_per_block(cfg, n_block_mult),
    )


# ---------------------------------------------------------------------------
# ``partial``-friendly feasibility steps: each takes the config positionally (so
# ``filter``/``map`` supply it) with the device limits / B / per-backend SMEM fn bound
# by ``functools.partial`` at the call site.
# ---------------------------------------------------------------------------


@beartype
def _coop_resident(
    cfg: dict, *, smem_fn: Callable, lim: DeviceLimits, B: int, n_block_mult: int
) -> bool:
    """
    Whether the coop problem-sized grid co-resides at this config's SMEM/register
    footprint (``n_block_mult`` = the gemm's gate-width multiplier: 4 fwd, 1 bwd).
    """
    return _coop_grid_resident(
        cfg, smem=smem_fn(cfg), lim=lim, B=B, n_block_mult=n_block_mult
    )


@beartype
def _persistent_resident(
    cfg: dict, *, smem_fn: Callable, lim: DeviceLimits, n_block_mult: int
) -> bool:
    """
    Whether the persistent kernel's FIXED ``blocks_per_sm * sms`` cooperative
    grid co-resides -- i.e. ``blocks_per_sm`` blocks fit one SM at this
    config's SMEM / register footprint.
    """
    grid = cfg["blocks_per_sm"] * lim.sms

    return grid_resident(
        grid=grid,
        threads=cfg["threads"],
        smem=smem_fn(cfg),
        lim=lim,
        regs_per_block=_regs_per_block(cfg, n_block_mult),
    )


@beartype
def _persistent_bps_variants(cfg: dict, *, B: int, sms: int) -> Iterator[dict]:
    """
    Add the ``blocks_per_sm`` variants to a persistent tile. Always offers 1
    and 2 (the intra-SM latency-hiding pair); offers HIGHER bps only at batches
    whose tile count needs a bigger cooperative grid to keep
    ``tiles_per_block`` low.
    """
    tiles = tilelang.cdiv(B, cfg["block_M"]) * tilelang.cdiv(cfg["H"], cfg["block_N"])
    hi = min(4, max(2, tilelang.cdiv(tiles, sms)))
    for bps in range(1, hi + 1):
        yield {**cfg, "blocks_per_sm": bps}


@beartype
def coop_fwd_configs(
    H: int,
    B: int,
    device: torch.device,
    dtype_lo: str,
    dtype_hi: str,
    save_gates: bool = False,
) -> list[dict]:
    """
    Feasible coop tiles for (H, B): SMEM must fit and the cooperative grid
    (cdiv(H, block_N) * cdiv(B, block_M)) must co-reside on the device.
    """
    smem_fn = coop_fwd_smem_bytes
    policy = GemmWarpPolicy.FullRow
    lim = device_limits(device)
    base = {"H": H, "dtype_lo": dtype_lo, "dtype_hi": dtype_hi}
    if save_gates:
        base["save_gates"] = True

    # Candidates
    configs = fwd_candidate_tiles(H, base)
    configs = map(lambda cfg: {**cfg, "fpad": _FLAGS_FPAD}, configs)

    # Static constraints
    configs = filter(threads_are_warp, configs)
    configs = filter(partial(mnk_atom_ok, n_block_mult=4), configs)
    configs = filter(partial(wmma_ok, policy=policy, n_block_mult=4), configs)
    configs = filter(_tile_ok, configs)
    configs = filter(_epilogue_reduction_ok, configs)
    # Launch constraints
    configs = filter(partial(smem_fits, smem_fn=smem_fn, lim=lim), configs)
    configs = filter(
        partial(_coop_resident, smem_fn=smem_fn, lim=lim, B=B, n_block_mult=4), configs
    )

    return list(configs)


@beartype
def persistent_fwd_configs(
    H: int,
    B: int,
    device: torch.device,
    dtype_lo: str,
    dtype_hi: str,
    save_gates: bool = False,
) -> list[dict]:
    """
    Feasible persistent tiles, each tagged with a ``blocks_per_sm``. The
    grid is ``blocks_per_sm * sm_num`` cooperative blocks; feasibility is
    batch-independent: SMEM must fit and the whole grid must co-reside.
    """
    smem_fn = persistent_fwd_smem_bytes
    policy = GemmWarpPolicy.FullRow
    lim = device_limits(device)
    base = {"H": H, "dtype_lo": dtype_lo, "dtype_hi": dtype_hi}
    if save_gates:
        base["save_gates"] = True

    # Candidates
    configs = fwd_candidate_tiles(H, base)
    configs = map(lambda cfg: {**cfg, "fpad": _FLAGS_FPAD}, configs)
    configs = chain.from_iterable(
        map(partial(_persistent_bps_variants, B=B, sms=lim.sms), configs)
    )

    # Static constraints
    configs = filter(threads_are_warp, configs)
    configs = filter(partial(mnk_atom_ok, n_block_mult=4), configs)
    configs = filter(partial(wmma_ok, policy=policy, n_block_mult=4), configs)
    configs = filter(_tile_ok, configs)
    configs = filter(_epilogue_reduction_ok, configs)
    # Launch constraints
    configs = filter(partial(smem_fits, smem_fn=smem_fn, lim=lim), configs)
    configs = filter(
        partial(_persistent_resident, smem_fn=smem_fn, lim=lim, n_block_mult=4), configs
    )

    return list(configs)


@beartype
def _add_bwd_policy(cfg: dict) -> Iterator[dict]:
    """
    Yield a backward tile tagged with its warp ``policy`` -- ``Square``
    (balanced) where legal, else ``FullCol``
    """
    for policy in (GemmWarpPolicy.Square, GemmWarpPolicy.FullCol):
        if wmma_ok(cfg, policy, n_block_mult=1):
            yield {**cfg, "policy": policy}
            return


@beartype
def _coop_bwd_flag_variants(cfg: dict) -> Iterator[dict]:
    """
    Add both ``wpf`` (wait-pre-fetch) variants to a coop-bwd tile.
    """
    for wpf in (False, True):
        yield {**cfg, "wpf": wpf, "fpad": _FLAGS_FPAD}


@beartype
def coop_bwd_configs(
    H: int, B: int, device: torch.device, dtype_lo: str, dtype_hi: str
) -> list[dict]:
    """
    Feasible ``lstm_coop_bwd`` tiles for (H, B). The kernel grid-syncs every
    step, so the launch is cooperative -- the whole cdiv(H,bN)*cdiv(B,bM) grid
    must co-reside (the last m-tile may be ragged; the kernel predicates it).
    """
    smem_fn = coop_bwd_smem_bytes
    lim = device_limits(device)
    base = {"H": H, "dtype_lo": dtype_lo, "dtype_hi": dtype_hi}

    # All candidates: the backward tile region x {wpf=False, wpf=True} flag builds.
    configs = bwd_candidate_tiles(H, base)
    configs = chain.from_iterable(map(_coop_bwd_flag_variants, configs))

    # Static constraints
    configs = filter(threads_are_warp, configs)
    configs = filter(partial(mnk_atom_ok, n_block_mult=1), configs)
    configs = chain.from_iterable(map(_add_bwd_policy, configs))
    configs = filter(_tile_ok, configs)
    # Launch constraints
    configs = filter(partial(smem_fits, smem_fn=smem_fn, lim=lim), configs)
    configs = filter(
        partial(_coop_resident, smem_fn=smem_fn, lim=lim, B=B, n_block_mult=1), configs
    )

    return list(configs)


@beartype
def persistent_bwd_configs(
    H: int, B: int, device: torch.device, dtype_lo: str, dtype_hi: str
) -> list[dict]:
    """
    Feasible ``lstm_persistent_bwd`` tiles. Like ``persistent_fwd_configs``
    (fwd): the grid is the fixed ``blocks_per_sm * sm_num`` cooperative blocks,
    so feasibility is batch-independent -- SMEM must fit and the whole grid
    must co-reside.
    """
    smem_fn = persistent_bwd_smem_bytes
    lim = device_limits(device)
    base = {"H": H, "dtype_lo": dtype_lo, "dtype_hi": dtype_hi}

    # All candidates
    configs = bwd_candidate_tiles(H, base)
    configs = map(lambda cfg: {**cfg, "fpad": _FLAGS_FPAD}, configs)
    configs = chain.from_iterable(
        map(partial(_persistent_bps_variants, B=B, sms=lim.sms), configs)
    )

    # Static constraints
    configs = filter(threads_are_warp, configs)
    configs = filter(partial(mnk_atom_ok, n_block_mult=1), configs)
    configs = chain.from_iterable(map(_add_bwd_policy, configs))
    configs = filter(_tile_ok, configs)
    # Launch constraints
    configs = filter(partial(smem_fits, smem_fn=smem_fn, lim=lim), configs)
    configs = filter(
        partial(_persistent_resident, smem_fn=smem_fn, lim=lim, n_block_mult=1), configs
    )

    return list(configs)


@beartype
def splitk_bwd_configs(
    H: int, B: int, device: torch.device, dtype_lo: str, dtype_hi: str
) -> list[dict]:
    """
    Feasible ``lstm_coop_bwd_splitk`` tiles across all swept ``k_splits`` (each
    config carries its own), each tagged with the largest feasible ``res_kc``
    (resident strip prefix).
    """
    lim = device_limits(device)
    base = {"H": H, "dtype_lo": dtype_lo, "dtype_hi": dtype_hi}

    # All candidates
    configs = splitk_candidate_tiles(H, base)

    # Static constraints (split-K runs the Square gemm; n_block_mult=1 -- its gemm emits a
    # [block_M, block_N] tile, the gates are formed in the epilogue, not the gemm).
    configs = filter(threads_are_warp, configs)
    configs = filter(partial(mnk_atom_ok, n_block_mult=1), configs)
    configs = filter(
        partial(wmma_ok, policy=GemmWarpPolicy.Square, n_block_mult=1), configs
    )
    configs = filter(_tile_ok, configs)
    configs = filter(_splitk_extra_ok, configs)
    configs = filter(_splitk_store_layout_ok, configs)

    # Launch constraints: tag the largest res_kc whose SMEM fits a block AND lets the
    # cdiv(H, block_N) * k_splits grid co-reside (the tile is dropped if none fits).
    configs = chain.from_iterable(map(partial(_add_splitk_res_kc, lim=lim), configs))

    return list(configs)


@beartype
def _splitk_extra_ok(cfg: dict) -> bool:
    """
    Split-K's EXTRA feasibility constraints beyond the shared ``mnk_atom_ok`` /
    ``_tile_ok``:

      1. ``k_splits`` must split the flat-K reduction ``4H`` EVENLY (the kernel asserts
         ``(4H) % k_splits == 0``). This matters now that ``k_splits`` carries non-pow2
         values that divide only some H -- e.g. ks=3 divides 4*1536=6144 but not
         4*1024=4096, so it is feasible at H=1536 and pruned at H=1024.
      2. The per-block flat-K range ``4H/k_splits`` must be a whole number of ``block_K``
         chunks. NOT implied by ``_tile_ok``'s ``H % block_K`` once block_K has non-pow2
         values (e.g. H=1536, ks=32: k_range=192 admits bK=96 with H % 96 == 0).
    """
    flat_k = 4 * cfg["H"]
    if flat_k % cfg["k_splits"] != 0:
        return False
    k_range = flat_k // cfg["k_splits"]  # flat-K extent owned by one block
    return k_range % cfg["block_K"] == 0


@beartype
def _splitk_store_layout_ok(cfg: dict) -> bool:
    """
    Split-K-only: whether the phase-1 MMA accumulator fragment admits a layout for its
    DIRECT register->global store.

    Unlike coop/persistent backward (which route the [block_M, block_N] accumulator through
    a gate epilogue), split-K phase-1 copies it straight to the global partial-sum scratch
    ``P`` (``T.copy(dh_local, P[...])`` in ``lstm_coop_bwd_splitk``). TileLang's layout
    inference cannot find a store layout for that m16n8 C-fragment when the Square warp
    split STACKS rows (m_warp >= 2) AND each warp owns an ODD number (>= 3) of 8-wide
    n-atoms: the leftover unpaired atom has no placement.
    """
    m_warp, n_warp = GemmWarpPolicy.Square.compute_warp_partition(
        cfg["block_M"], cfg["block_N"], cfg["threads"] // TPW
    )

    atoms_per_warp = (cfg["block_N"] // n_warp) // 8  # 8-wide n-atoms per warp

    return m_warp == 1 or atoms_per_warp == 1 or atoms_per_warp % 2 == 0


@beartype
def _add_splitk_res_kc(cfg: dict, *, lim: DeviceLimits) -> Iterator[dict]:
    """
    Yield the split-K tile tagged with the LARGEST feasible ``res_kc``
    (resident strip prefix), or nothing if none fits.
    """
    bN, bK, threads = cfg["block_N"], cfg["block_K"], cfg["threads"]
    H, k_splits = cfg["H"], cfg["k_splits"]
    kchunks = (4 * H // k_splits) // bK
    grid = tilelang.cdiv(H, bN) * k_splits

    for r in range(kchunks, 0, -1):
        # SMEM bytes from the single source of truth (configs_smem); the search just picks
        # the largest res_kc whose SMEM both fits a block and keeps the grid resident.
        smem = splitk_smem_bytes_at(cfg, r)
        if smem > lim.smem_per_block:
            continue

        if not grid_resident(
            grid=grid,
            threads=threads,
            smem=smem,
            lim=lim,
            regs_per_block=_regs_per_block(cfg, 1),
        ):
            continue

        yield {**cfg, "res_kc": r}
        return
