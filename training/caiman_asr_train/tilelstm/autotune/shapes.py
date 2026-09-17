from beartype import beartype

from caiman_asr_train.utils.math import round_up

# Representative sequence length for tuning/benchmarking. The scan cost is linear
# in S and the coop/persistent crossover is set by batch-vs-hidden residency, not
# S, so a fixed proxy is enough; the tuned kernel is dynamic in S regardless.
PROFILE_SEQ = 64


@beartype
def _ceil_power_of_two(n: int) -> int:
    if n < 1:
        raise ValueError("n must be positive")
    return 1 << (n - 1).bit_length()


@beartype
def tune_batch_for(B: int) -> int:
    """
    Tune-time batch bucket for ``B``.

    Bands:
      B <= 32   → next power of two   (fine-grained for very small batch)
      B > 32    → next multiple of 16 (uniform fine resolution across the batch range)
    """
    if B <= 32:
        return _ceil_power_of_two(B)

    return round_up(B, multiple_of=16)
