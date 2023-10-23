#!/usr/bin/env python3
from editdistance import eval as levenshtein
from hypothesis import given
from hypothesis.strategies import text


@given(text(), text())
def test_levenshtein1(a, b):
    assert levenshtein(a, b) == levenshtein_ref(a, b)
    assert levenshtein(a.split(), b.split()) == levenshtein_ref(a.split(), b.split())


def test_levenshtein2():
    assert (
        levenshtein("Mary had a little lamb".split(), "Mary had a little lamb".split())
        == 0
    )
    assert levenshtein("I have a pet dog".split(), "You have a pet cat".split()) == 2
    assert levenshtein("one two three".split(), "two three".split()) == 1
    assert levenshtein("one two three".split(), "two three one".split()) == 2


def levenshtein_ref(a, b):
    """Calculates the Levenshtein distance between two sequences."""

    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]
