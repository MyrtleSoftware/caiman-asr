import pytest
from beartype import beartype

from caiman_asr_train.keywords.trie import Keywords


def near(a, b, tol=1e-6):
    return abs(a - b) < tol


@pytest.mark.parametrize(
    "keywords, seq, expected_delta",
    [
        # Null input
        ([], "", []),
        # Single keyword success
        ([("ab", 1.0)], "cabc", [0, 1, 1, 0]),
        # Single keyword fail
        ([("abc", 1.0)], "cabx", [0, 1, 1, -2]),
        # Keyword as prefix
        ([("ab", 1.0), ("abc", 1.0)], "abx", [2, 2, -2]),
        ([("ab", 1.0), ("abcd", 1.0)], "abcx", [2, 2, 1, -3]),
        # Negative keyword
        ([("ab", -1.0)], "cabc", [0, -1, -1, 0]),
        # Overlapping keywords offset
        ([("ab", 1.0), ("bcd", -1.0)], "abcx", [1, 0, -1, 2]),
        # Overlapping keywords no-offset
        ([("ab", 1.0), ("abc", -1.0)], "abc", [0, 0, -1]),
        ([("ab", 2.0), ("abc", 1.0)], "abx", [3, 3, -2]),
        # Test spaces,
        ([(" ab", 1.0)], "a ab", [0, 1, 1, 1]),
    ],
)
@beartype
def test_keywords(
    keywords: list[tuple[str, float]],
    seq: str,
    expected_delta: list[int],
):
    kw = Keywords(keywords)

    print(kw)

    delta = []
    state = Keywords.init()
    for char in seq:
        step, state = kw.step(char, state)
        delta.append(step)

    assert all(
        near(a, b) for a, b in zip(delta, expected_delta, strict=True)
    ), f"{delta=} != {expected_delta=}"

    state = Keywords.init()
    sum_delta, _ = kw.steps(seq, state)
    sum_expect = sum(expected_delta)
    assert near(sum_delta, sum_expect), f"{sum_delta=} != {sum_expect=}"


@pytest.mark.xfail
def test_duplicate():
    Keywords([("ab", 1.0), ("ab", 2.0)])
