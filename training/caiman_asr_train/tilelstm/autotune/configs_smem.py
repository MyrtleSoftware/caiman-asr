"""
Per-block shared-memory estimates for the TileLang LSTM kernels.

One ``*_smem_bytes`` function per backend, each reading the tile fields straight off
the config dict (so it composes with ``configs._smem_fits`` / the sweep's est_smem
column / the tune-time estimate validation).
"""

from beartype import beartype

from caiman_asr_train.autotune.dtypes import dtype_bytes


@beartype
def coop_fwd_smem_bytes(cfg: dict) -> int:
    """
    Peak shared-memory bytes for one ``lstm_coop_fwd`` block.

    The FullRow register epilogue keeps the gate gather + cell update + output
    hidden in fragments, and the cell accumulator (C_frag) is resident in registers
    too, so the ONLY shared buffers are the pipelined matmul operands (matmul phase)
    and the epilogue's prefetched gates (epilogue phase):

      Ys_shared   [block_M, block_K]      pipelined x num_stages  (matmul phase)
      rW_shared   [4*block_N, block_K]    pipelined x num_stages  (matmul phase)
      Gs_shared   [block_M, 4, block_N]   cp.async prefetch       (epilogue phase)

    Gs is prefetched *after* the gemm (as in the persistent kernel), so the
    aggressive SMEM-merge overlaps it onto the now-dead operand tiles -> the peak is
    the LARGER of the two phases, not their sum. Coop differs from persistent only in
    having no Cp_shared term: it keeps the prior cell resident in registers instead
    of re-reading it from global.

    ``save_gates`` adds the training forward's ``As_shared`` activation slab
    ([block_M, 4, block_N] lo) to the epilogue phase -- same lifetime as Gs_shared,
    so it is part of the same merged-overlap peak, not added on top of the matmul.
    """
    bM, bN, bK, ns = cfg["block_M"], cfg["block_N"], cfg["block_K"], cfg["num_stages"]
    lo = dtype_bytes(cfg, "dtype_lo")
    save_gates = cfg.get("save_gates", False)
    # Ys_shared + rW_shared
    matmul = ns * (bM * bK * lo + 4 * bN * bK * lo)
    # Gs_shared (+ As_shared when save_gates); cell state is resident in registers.
    epilogue = bM * 4 * bN * lo * (2 if save_gates else 1)
    return max(matmul, epilogue)


@beartype
def persistent_fwd_smem_bytes(cfg: dict) -> int:
    """
    Peak shared-memory bytes for one ``lstm_persistent_fwd`` block.

    The FullRow register epilogue keeps the gate gather + cell update + output in
    fragments (registers), so the ONLY shared buffers are the pipelined matmul
    operands (matmul phase) and the epilogue's prefetched Gs/Cp (epilogue phase).
    The Gs/Cp prefetch is issued after the K-loop, so the aggressive SMEM-merge
    overlaps them onto the now-dead operand tiles -> the peak is the LARGER of the
    two phases, not their sum.

    ``save_gates`` adds the training forward's ``As_shared`` activation slab
    ([block_M, 4, block_N] lo) to the epilogue phase -- same lifetime as Gs_shared,
    so part of the merged-overlap peak, not added on top of the matmul.
    """
    bM, bN, bK, ns = cfg["block_M"], cfg["block_N"], cfg["block_K"], cfg["num_stages"]
    lo, hi = dtype_bytes(cfg, "dtype_lo"), dtype_bytes(cfg, "dtype_hi")
    save_gates = cfg.get("save_gates", False)
    # Ys_shared + rW_shared (4 gates)
    matmul = ns * (bM * bK * lo + 4 * bN * bK * lo)
    # Gs_shared + Cp_shared (+ As_shared when save_gates).
    epilogue = bM * 4 * bN * lo * (2 if save_gates else 1) + bM * bN * hi
    return max(matmul, epilogue)


@beartype
def coop_bwd_smem_bytes(cfg: dict) -> int:
    """
    Peak shared-memory bytes for one ``lstm_coop_bwd`` block.

    The rebuilt real kernel keeps the gate epilogue entirely in registers with
    direct-global input reads (see its docstring), so the base shared buffers
    are the two pipelined matmul operands:
      dG_shared   [block_M, block_K]   pipelined x num_stages  (matmul phase)
      rWb_shared  [block_K, block_N]   pipelined x num_stages  (matmul phase)

    ``wpf`` adds the step-top ``As_sh`` stage ([4, block_M, block_N] lo); unlike
    the post-gemm forward prefetch this is a step-top load held live ACROSS the
    gemm phase, so it sits on top of the matmul stages rather than merging onto
    a dead tile -- it is added to the peak, not max'd.
    """
    bM, bN, bK, ns = cfg["block_M"], cfg["block_N"], cfg["block_K"], cfg["num_stages"]
    lo = dtype_bytes(cfg, "dtype_lo")
    # 1*bK-wide dG_shared + rWb_shared; no resident carry, no epilogue phase.
    base = ns * (bM * bK * lo + bK * bN * lo)
    if cfg.get("wpf", False):
        base += 4 * bM * bN * lo  # As_sh, live across the gemm
    return base


@beartype
def persistent_bwd_smem_bytes(cfg: dict) -> int:
    """
    Peak shared-memory bytes for one ``lstm_persistent_bwd`` block.

    ``lstm_persistent_bwd`` is jitted with the aggressive SMEM-merge ON
    (``LSTM_PASS_CONFIGS``): the FIVE epilogue buffers (A/Cprev/Ccur/dCin/dh_shared)
    are dead during the matmul and the two operand tiles (dG_shared/rWb_shared) are
    dead during the epilogue, so the merge overlaps the epilogue onto the operand
    SMEM -> the peak is the ``max`` of the two phases, not their sum.

    matmul phase = num_stages * (dG_shared + rWb_shared), both 1*bK-wide native
    operands; epilogue phase = A_shared + Cprev + Ccur + dCin + dh_shared. ``Cprev`` /
    ``Ccur`` stage the saved cell, stored fp16 (``Cs`` is ``dtype_lo``), so they are
    lo-width; only ``dCin`` (the dc_fut carry, round-tripping global ``dC``) and
    ``dh_shared`` stay hi.
    """
    bM, bN, bK, ns = cfg["block_M"], cfg["block_N"], cfg["block_K"], cfg["num_stages"]
    lo, hi = dtype_bytes(cfg, "dtype_lo"), dtype_bytes(cfg, "dtype_hi")
    # 1*bK-wide dG_shared + rWb_shared
    matmul_phase = ns * (bM * bK * lo + bK * bN * lo)
    # A_shared + Cprev(lo) + Ccur(lo) + dCin(hi) + dh_shared(hi), merged onto operands.
    epilogue_phase = bM * 4 * bN * lo + 2 * bM * bN * lo + 2 * bM * bN * hi
    return max(matmul_phase, epilogue_phase)


@beartype
def splitk_smem_bytes_at(cfg: dict, res_kc: int) -> int:
    """
    Single source of truth for one ``lstm_coop_bwd_splitk`` block's dynamic SMEM (bytes) at
    a given ``res_kc`` (resident strip prefix): the strip + the ns-deep dG pipeline, plus
    the streamed tail's ns-deep rW pipeline when the strip is PARTIAL (``res_kc < kchunks``;
    the tail reuses dG_shared, so only the rW slab adds).
    """
    bM, bN, bK, ns = cfg["block_M"], cfg["block_N"], cfg["block_K"], cfg["num_stages"]
    lo = dtype_bytes(cfg, "dtype_lo")
    kchunks = (4 * cfg["H"] // cfg["k_splits"]) // bK
    smem = res_kc * bK * bN * lo + ns * bM * bK * lo
    if res_kc < kchunks:
        smem += ns * bK * bN * lo  # streamed-tail rW pipeline (reuses dG_shared)
    return smem


@beartype
def splitk_smem_bytes(cfg: dict) -> int:
    """
    Per-block dynamic SMEM for a ``lstm_coop_bwd_splitk`` config -- the cfg-level wrapper
    over ``splitk_smem_bytes_at`` (the single source of truth shared with the candidate
    search in ``configs._add_splitk_res_kc``). Used by the tune-time estimate validation
    (``sweep.autotune_select``) and the sweep harness's est_smem column.
    """
    return splitk_smem_bytes_at(cfg, cfg["res_kc"])
