import statistics
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from beartype import beartype


@beartype
@dataclass(frozen=True)
class ExcessStats:
    """
    Finer-grained stats for a set of K values (each K >= 1; lower better), where the
    excess depth is ``e = K - 1``:

    ``n``          number of cells
    ``sum_abs``    Σe
    ``sum_sq``     Σe²  (heavy-tail sensitive)
    ``mean``       Σe / n
    ``rms``        sqrt(Σe² / n)
    ``p_perfect``  fraction of cells with K == 1 (a perfect first hit)
    ``medK``       median K
    ``maxK``       worst (largest) K
    """

    n: int = 0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    mean: float = 0.0
    rms: float = 0.0
    p_perfect: float = 0.0
    medK: float = 0.0
    maxK: float = 0.0


@beartype
def excess_stats(ks: Iterable[float]) -> ExcessStats:
    """
    The :class:`ExcessStats` for an iterable of K values (all-zero when empty).
    """
    ks = list(ks)
    n = len(ks)
    if not n:
        return ExcessStats()
    exc = [k - 1.0 for k in ks]
    sq = sum(e * e for e in exc)
    s1 = sum(exc)
    return ExcessStats(
        n=n,
        sum_abs=s1,
        sum_sq=sq,
        mean=s1 / n,
        rms=(sq / n) ** 0.5,
        p_perfect=sum(1 for k in ks if k <= 1.0 + 1e-9) / n,
        medK=float(statistics.median(ks)),
        maxK=float(max(ks)),
    )


@beartype
def k_metric(preds: np.ndarray, lat: np.ndarray, best: float, tol: float) -> float:
    """
    Min tie-aware 1-based rank (by predicted speed) over configs measured within ``tol``
    of ``best``. Lower = the model surfaces a near-optimal config sooner.
    """
    within = np.nonzero(lat <= best * tol)[0]
    # Non-empty whenever best == lat.min() (the argmin is within any tol >= 1); an empty
    # set means best/lat disagree -- e.g. NaN latencies -- and min(ks) would raise bare.
    assert within.size, f"no latency within {tol}x of best={best}; NaN in the ranking?"
    ks = []
    for i in within:
        p = preds[i]
        strictly = int(np.sum(preds < p))
        ties = int(np.sum(preds == p))
        ks.append(strictly + (ties + 1) / 2.0)
    return min(ks)
