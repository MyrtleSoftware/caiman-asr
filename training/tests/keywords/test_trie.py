import math

import pytest

from caiman_asr_train.keywords.trie import Keywords


def near(a, b, tol=1e-6):
    return abs(a - b) < tol


def shape(depth: int, c0: float = 0.3, beta: float = 0.9) -> float:
    if depth <= 0:
        return 0.0
    if depth == 1:
        return c0
    return c0 * beta + math.log(depth)


@pytest.mark.parametrize(
    "keywords, seq, expected_deltas_fn",
    [
        # 1) Null input
        ([], "", lambda s1, s2, s3: []),
        # 2) Single keyword success: ["ab"], "cabc"
        #    deltas: [0, s1, s2, 0]
        (["ab"], "cabc", lambda s1, s2, s3: [0.0, s1, s2, 0.0]),
        # 3) Single keyword incomplete + refund: ["abc"], "cabx"
        #    deltas: [0, s1, s2, -s2]
        (["abc"], "cabx", lambda s1, s2, s3: [0.0, s1, s2, -s2]),
        # 4) Keyword as prefix: ["ab", "abc"], "abx"
        #    Here 'ab' is terminal; on mismatch we do NOT refund.
        #    deltas: [s1, s2, 0]
        (["ab", "abc"], "abx", lambda s1, s2, s3: [s1, s2, 0.0]),
        # 5) Prefix of longer word (non-terminal at depth 3): ["ab", "abcd"], "abcx"
        #    'abc' is not terminal, so the last step refunds s3
        #    deltas: [s1, s2, s3, -s3]
        (["ab", "abcd"], "abcx", lambda s1, s2, s3: [s1, s2, s3, -s3]),
        # 6) Leading space in keyword: [" ab"], "a ab"
        #    match starts at the space; deltas: [0, s1, s2, s3]
        ([" ab"], "a ab", lambda s1, s2, s3: [0.0, s1, s2, s3]),
    ],
)
def test_keywords_depth_potential_behaviour(keywords, seq, expected_deltas_fn):
    # Use module defaults (c0=0.3, beta=0.9, context_score=1.0)
    kw = Keywords(keywords)

    # Pre-compute shaped steps
    s1 = shape(1)
    s2 = shape(2)
    s3 = shape(3)

    expected_delta = expected_deltas_fn(s1, s2, s3)

    # Step-by-step
    delta = []
    state = Keywords.init()
    for ch in seq:
        d, state = kw.step(ch, state)
        delta.append(d)

    assert len(delta) == len(expected_delta)
    assert all(
        near(a, b) for a, b in zip(delta, expected_delta, strict=True)
    ), f"{delta=} != {expected_delta=}"

    # Aggregate steps() must equal sum of per-step
    state = Keywords.init()
    sum_delta, _ = kw.steps(seq, state)
    assert near(
        sum_delta, sum(expected_delta)
    ), f"{sum_delta=} != {sum(expected_delta)=}"


@pytest.mark.xfail
def test_duplicate():
    Keywords(["ab", "ab"])
