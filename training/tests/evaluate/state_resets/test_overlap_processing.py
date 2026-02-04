import pytest

from caiman_asr_train.evaluate.state_resets.overlap_processing import (
    best_split,
    combine_predictions,
    denorm,
    get_unique_predictions,
    process_time,
)


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            ([[2, 6, 7, 15], [1, 3, 10, 11]], 2, 267, 67),
            ([2, 6, 7, 15, 1 + 100, 3 + 100, 10 + 100, 11 + 100]),
        ),
        (
            ([[2, 6], [1, 3], [7, 10]], 2, 267, 67),
            ([2, 6, 1 + 100, 3 + 100, 7 + 2 * 100, 10 + 2 * 100]),
        ),
        (
            ([[2, 6], [1, 3], [7, 10], [1, 2]], 2, 267, 67),
            (
                [
                    2,
                    6,
                    1 + 100,
                    3 + 100,
                    7 + 2 * 100,
                    10 + 2 * 100,
                    1 + 3 * 100,
                    2 + 3 * 100,
                ]
            ),
        ),
    ],
)
def test_process_time(test_input, expected):
    returned = process_time(test_input[0], test_input[1], test_input[2], test_input[3])
    assert returned == expected


@pytest.mark.parametrize(
    """pred, expected_pred""",
    [
        (
            [[1, 2, 3, 4], [4, 5, 6, 7]],
            [[1, 2, 3, 4], [5, 6, 7]],
        ),
        (
            [[1, 2, 3, 4], [4, 4, 6, 7]],
            [[1, 2, 3, 4], [4, 6, 7]],
        ),
        (
            [[], [1, 2, 3, 4], [4, 4, 6, 7]],
            [[], [1, 2, 3, 4], [4, 6, 7]],
        ),
        (
            [[7, 2, 3, 6, 5], [2, 6, 5, 9, 7]],
            [[7, 2, 3, 6, 5], [9, 7]],
        ),
        (
            [[7, 2, 3, 6, 5], [2, 6, 5, 9, 7]],
            [[7, 2, 3, 6, 5], [9, 7]],
        ),
    ],
)
def test_get_unique_predictions(
    pred,
    expected_pred,
):
    timestamps = pred[:]
    probs = [list(map(float, p)) for p in pred]

    expected_timestamps = expected_pred[:]
    expected_probs = [list(map(float, p)) for p in expected_pred]

    adjusted_pred, adjusted_timestamps, adjusted_probs = get_unique_predictions(
        pred, timestamps, probs, detokenize=lambda x: str(x) + str(x)
    )

    for i, segment in enumerate(adjusted_pred):
        assert len(segment) == len(adjusted_timestamps[i])
    assert adjusted_pred == expected_pred
    assert adjusted_timestamps == expected_timestamps
    assert adjusted_probs == expected_probs


@pytest.mark.parametrize(
    "ex_input, expected",
    [
        ([[0, 1], [3], [6, 7, 9, 9]], [[0, 1, 3, 6, 7, 9, 9]]),
        ([[0, 1], [], [6, 7, 9, 9]], [[0, 1, 6, 7, 9, 9]]),
        ([[0, 1], [6, 7, 9, 9]], [[0, 1, 6, 7, 9, 9]]),
        (
            [[0.98, 0.76, 0.33], [0.11, 0.98, 0.99]],
            [[0.98, 0.76, 0.33, 0.11, 0.98, 0.99]],
        ),
        ([[0.99, 0.99, 0.99]], [[0.99, 0.99, 0.99]]),
    ],
)
def test_combine_predictions(ex_input, expected):
    returned = combine_predictions(ex_input)
    assert expected == returned


@pytest.mark.parametrize(
    "lhs, rhs, expect",
    [
        (["the", "cat", "sat"], ["sat", "on", "a"], (1, 0, 0, 1)),
        (["the", "cat", "sat"], ["cat", "sat", "on", "a"], (2, 0, 0, 2)),
        (["the", "cat", "flap"], ["cat", "flip", "a"], (2, 0, 0, 2)),
    ],
)
def test_best_split(lhs: list[str], rhs: list[str], expect):
    #
    ll = list(enumerate(lhs))
    rr = list(enumerate(rhs))

    _, n, m, x, y = best_split(ll, rr)

    assert (n, m, x, y) == expect, f"Expected {expect}, got {(n, m, x, y)=}"


@pytest.mark.parametrize(
    "input, expect",
    [
        ("ooh", "ooh"),
        ("a!", "a"),
        ("A!", "a"),
        ("!", ""),
        ("<enen>", ""),
    ],
)
def test_denorm(input: str, expect: str):
    assert denorm(input, detokenize=lambda x: x) == expect
