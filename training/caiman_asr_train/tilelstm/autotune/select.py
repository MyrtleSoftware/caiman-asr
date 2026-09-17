from collections.abc import Callable
from functools import cache

import torch
from beartype import beartype

from caiman_asr_train.tilelstm.autotune import tuners
from caiman_asr_train.tilelstm.autotune.kernel import Kernel
from caiman_asr_train.tilelstm.autotune.shapes import tune_batch_for
from caiman_asr_train.tilelstm.tree_model.core.prune import make_tree_ranker

# Split-K is wins ONLY at small batch
_SPLITK_MAX_BATCH = 384

# Keep this many tree-predicted-fastest configs per (backend, shape) before the compile/
# bench loop; split-K keeps more, its config space being ~10x larger.
_TREE_TOPK = 50
_TREE_TOPK_SPLITK = 75


@cache
@beartype
def select_fastest(
    backends: tuple[Kernel, ...],
    H: int,
    B: int,
    device: torch.device,
    lo: torch.dtype,
    hi: torch.dtype,
) -> Callable:
    """
    Tune every backend that can build/launch at the shape and return the FASTEST -- the
    single source of truth for the backend-selection policy shared by the inference
    forward, the gate-saving training forward, and the recurrent backward.

    * **tree config-ranker** -- each backend's candidate set is pruned to the top-K
      tree-predicted-fastest configs before compile/bench (``make_tree_ranker``).

    * **split-K only at small batch** -- ``splitk_bwd`` is skipped at
      ``tune_batch >= _SPLITK_MAX_BATCH`` (batch-stable; only wins below that, a dense
      kernel above). The cutoff is tested against the tuned bucket, not raw ``B``, so the
      skip decision matches the batch the kernel is actually benched/cached at.

    * **persistent is a coop fallback** -- a ``persistent*`` backend is skipped once a
      ``coop*`` backend has launched, and is benched only when coop could not launch.
    """
    tune_batch = tune_batch_for(B)
    ranker = make_tree_ranker(
        k=_TREE_TOPK, k_splitk=_TREE_TOPK_SPLITK, device=device, verbose=False
    )
    best_kernel, best_ms = None, float("inf")
    coop_launched = False

    # Coop before split-K before persistent: coop-feasibility decides the
    # persistent skip, so coop MUST be benched first whatever order the caller
    # passed.
    for backend in sorted(
        backends, key=lambda kk: (kk.is_persistent(), kk.is_splitk())
    ):
        # Split-K only pays off at small batch -> skip it once the tuned bucket is
        # large (tune_batch, not raw B: it is the batch the kernel is benched at).
        if backend.is_splitk() and tune_batch >= _SPLITK_MAX_BATCH:
            continue

        # Persistent is the coop-can't-launch fallback: if a coop backend already launched,
        # persistent is not worth the tune, so skip it.
        if backend.is_persistent() and coop_launched:
            continue

        # tune returns None for an expected infeasibility (no config runs at this shape);
        # a correctness AssertionError / unexpected error propagates out of tune.
        tuned = tuners.tune(
            backend, H, device, tune_batch, lo, hi, config_ranker=ranker
        )

        if tuned is None:
            # sweep.autotune_select already logged this backend's "no feasible config";
            # don't repeat it here.
            continue

        kernel, ranked = tuned
        ms = ranked[0]["latency_ms"]

        if backend.is_coop():
            coop_launched = True

        if ms < best_ms:
            best_kernel, best_ms = kernel, ms

    assert best_kernel is not None, f"no candidate kernel launched for H={H} B={B}"

    return best_kernel


@beartype
def select_fwd_kernel(
    H: int, B: int, device: torch.device, lo_dtype: torch.dtype, hi_dtype: torch.dtype
) -> Callable:
    """
    The fastest tuned forward-scan kernel for ``(H, B, dtypes)`` on ``device``.
    """
    return select_fastest(
        (Kernel.fwd_c, Kernel.fwd_p), H, B, device, lo_dtype, hi_dtype
    )


@beartype
def select_fwd_train_kernel(
    H: int, B: int, device: torch.device, lo_dtype: torch.dtype, hi_dtype: torch.dtype
) -> Callable:
    """
    The fastest tuned gate-saving training forward for ``(H, B, dtypes)`` on ``device``.
    """
    return select_fastest(
        (Kernel.fwd_ct, Kernel.fwd_pt), H, B, device, lo_dtype, hi_dtype
    )


@beartype
def select_bwd_train_kernel(
    H: int, B: int, device: torch.device, lo_dtype: torch.dtype, hi_dtype: torch.dtype
) -> Callable:
    """
    The fastest tuned recurrent backward for ``(H, B, dtypes)`` on ``device``.
    """
    return select_fastest(
        (Kernel.bwd_c, Kernel.bwd_s, Kernel.bwd_p), H, B, device, lo_dtype, hi_dtype
    )
