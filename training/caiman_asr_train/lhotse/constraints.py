from bisect import bisect_left

import numpy as np
from beartype import beartype
from beartype.typing import Any, Optional
from lhotse import Seconds
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import SamplingConstraint

from caiman_asr_train.lhotse.config import AreaBatch


@beartype
class AreaConstraint(SamplingConstraint):
    def __init__(
        self,
        area_bins: list[tuple[float, int]],
        area: AreaBatch,
        strict: bool,
    ) -> None:
        """
        Represents a time * token based constraint for sampler classes.

        For a description of strict see the documentation of _find_smallest_bucket.
        """

        # Static
        self.bins = np.asarray(sorted(area_bins))
        self.area_limit = area
        self.strict = strict

        # State
        self.current_bucket: Optional[int] = None
        self.area: float = 0.0
        self.max_U: int = 0
        self.max_T: float = 0.0
        self.num_cuts: int = 0

    def add(self, cut: Cut) -> None:
        """
        Update the sampling constraint with the information about the sampled
        example (e.g. current batch size, num tokens, total duration).
        """
        dur, tok = self._length(cut)

        bucket_idx = _find_smallest_bucket(self.bins, dur, tok, strict=self.strict)

        assert bucket_idx is not None, "Cut does not fit in any bucket!"

        if self.current_bucket is None:
            self.current_bucket = bucket_idx
        else:
            assert self.current_bucket == bucket_idx, "User error: mixing buckets!"

        self.area += dur * tok
        self.max_U = max(self.max_U, tok)
        self.max_T = max(self.max_T, dur)
        self.num_cuts += 1

    def exceeded(self) -> bool:
        """
        Inform if the sampling constraint has been exceeded.
        """
        return self.area_limit.full(
            batch_size=self.num_cuts,
            max_U=self.max_U,
            max_T=self.max_T,
            sum_ut=self.area,
        )

    def close_to_exceeding(self) -> bool:
        """
        Check if the batch is close to satisfying the constraints. We define
        "closeness" as: if we added one more cut that has duration/tokens equal
        to the biggest seen cut in the current batch, then the batch would have
        exceeded the constraints.
        """
        return self.area_limit.full(
            batch_size=self.num_cuts + 1,
            max_U=self.max_U,
            max_T=self.max_T,
            sum_ut=self.area + self.max_U * self.max_T,
        )

    def reset(self) -> None:
        """
        Resets the internal state (called after yielding a mini-batch).
        """
        self.current_bucket = None
        self.area = 0.0
        self.max_U = 0
        self.max_T = 0.0
        self.num_cuts = 0

    def measure_length(self, example: Any) -> float:
        """
        Not implemented--area is only weakly ordered.
        """
        raise NotImplementedError("Area is only weakly ordered")

    def select_bucket(
        self, buckets: Any, example: Any, example_len: Any = None
    ) -> Optional[int]:
        """
        May fail if no valid bucket.
        """
        assert example_len is None
        return self._select_bucket(example)

    def _select_bucket(self, example: Cut) -> Optional[int]:
        dur, tok = self._length(example)
        return _find_smallest_bucket(self.bins, dur, tok, strict=self.strict)

    def _length(self, cut: Cut) -> tuple[Seconds, int]:
        return cut.duration, sum(
            len(seg.tokens) for seg in cut.supervisions if seg.has_custom("tokens")
        )


@beartype
class BucketingFilter:
    def __init__(self, constraint: AreaConstraint) -> None:
        """
        Filters out examples that did not fit into any of the buckets.
        Intended mainly for 2D bucketing.
        """
        self.constraint = constraint

    def __call__(self, cut: Cut) -> bool:
        return self.constraint._select_bucket(cut) is not None


@beartype
class SingleBucketFilter:
    def __init__(self, constraint: AreaConstraint, bucket_idx: int) -> None:
        """
        Filters out examples that do not fit into the specified bucket.
        Intended mainly for 2D bucketing.
        """
        self.constraint = constraint
        self.bucket_idx = bucket_idx

        assert 0 <= bucket_idx < len(constraint.bins)

    def __call__(self, cut: Cut) -> bool:
        if (b := self.constraint._select_bucket(cut)) is None:
            return False
        else:
            return b == self.bucket_idx


def _find_smallest_bucket(
    buckets: np.ndarray,
    duration: float,
    tokens: int,
    strict: bool = True,
) -> Optional[int]:
    """
    Find the smallest bucket that fits a given example.

    Arguments:
        buckets: array with axis [dur_buckets, token_buckets]
        duration: duration of the example
        tokens: number of tokens in the example
        strict: whether to use strict or lenient 2D bucketing (see below)


    2D bucketing 'strict' mode: only consider sub-buckets (token-buckets) for
    the specific bucket that matches this duration.

    E.g. for buckets = [(10, 5), (10, 10), (20, 12), (20, 18)]
         and (duration, tokens) = (8, 11)
         we will return None because we only consider the first two buckets
         based on dim0 (=8).

    2D bucketing 'lenient' mode - linear search (as 2nd dim may not be growing
    monotonically).

    E.g. for buckets = [(10, 5), (10, 10), (20, 12), (20, 18)]
         and example_lens = (8, 11)
         we will return bucket_idx=2 because (20, 12) fits (8, 11) at the cost padding.
    """

    if strict:
        # Sentinel
        n = buckets.shape[0]

        # Find the first 2D bucket that accepts this example
        dim0_begin = bisect_left(buckets[:, 0], duration)
        if dim0_begin == n:
            return None

        # Find the last 2D bucket that accepts this example
        dim0_end = dim0_begin
        while dim0_end < n and buckets[dim0_end, 0] == buckets[dim0_begin, 0]:
            dim0_end += 1

        # Find the smallest 2D bucket in this range that accepts this example
        dim1_begin = bisect_left(buckets[dim0_begin:dim0_end, 1], tokens)
        if dim1_begin == dim0_end - dim0_begin:
            return None

        return dim0_begin + dim1_begin

    does_fit = np.all(np.asarray([duration, tokens]) <= buckets, axis=1)
    min_fit_idx = np.argmax(does_fit)

    if min_fit_idx or does_fit[min_fit_idx]:
        return min_fit_idx.item()

    return None
