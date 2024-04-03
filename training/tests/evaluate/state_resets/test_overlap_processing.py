import pytest

from caiman_asr_train.evaluate.state_resets.overlap_processing import (
    combine_predictions,
    get_unique_predictions,
    manage_boundary_common_tokens,
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
    "pred, timestamps, enc_time, overlap, lookahead, expected_pred, expected_timestamps",
    [
        (
            [[1, 2, 3, 4], [4, 5, 6, 7]],
            [[20, 60, 100, 200], [10, 61, 101, 201]],
            2,
            67,
            3,
            [[1, 2, 3, 4], [5, 6, 7]],
            [[20, 60, 100, 200], [61, 101, 201]],
        ),
        (
            [[1, 2, 3, 4], [4, 4, 6, 7]],
            [[20, 60, 100, 200], [10, 61, 101, 201]],
            2,
            67,
            3,
            [[1, 2, 3, 4], [6, 7]],
            [[20, 60, 100, 200], [101, 201]],
        ),
        (
            [[], [1, 2, 3, 4], [4, 4, 6, 7]],
            [[], [40, 60, 100, 110], [35, 61, 101, 111]],
            2,
            67,
            3,
            [[], [1, 2, 3, 4], [4, 6, 7]],
            [[], [40, 60, 100, 110], [61, 101, 111]],
        ),
        (
            [[], [1, 2, 3, 4], [4, 4, 6, 7]],
            [[], [33, 60, 100, 110], [40, 61, 101, 111]],
            2,
            67,
            3,
            [[], [2, 3, 4], [4, 6, 7]],
            [[], [60, 100, 110], [61, 101, 111]],
        ),
        (
            [[7, 2, 3, 6, 5], [2, 6, 5, 9, 7]],
            [[1, 2, 3, 4, 6], [1, 3, 4, 5, 6]],
            1,
            2,
            3,
            [[7, 2, 3, 6, 5], [9, 7]],
            [[1, 2, 3, 4, 6], [5, 6]],
        ),
    ],
)
def test_get_unique_predictions(
    pred,
    timestamps,
    enc_time,
    overlap,
    lookahead,
    expected_pred,
    expected_timestamps,
):
    adjusted_pred, adjusted_timestamps = get_unique_predictions(
        pred, timestamps, enc_time, overlap, lookahead
    )
    for i, segment in enumerate(adjusted_pred):
        assert len(segment) == len(adjusted_timestamps[i])
    assert adjusted_pred == expected_pred
    assert adjusted_timestamps == expected_timestamps


@pytest.mark.parametrize(
    "segm, t_st, trusted_list, lookahead, exp_segm, exp_time",
    [
        (
            [7, 5, 9, 7, 6],
            [0, 1, 2, 3, 4],
            [7],
            1,
            [5, 9, 7, 6],
            [1, 2, 3, 4],
        ),
        ([7, 5, 9, 7, 6], [0, 1, 2, 3, 4], [7, 5, 9], 3, [7, 6], [3, 4]),
        ([7, 5, 9, 7, 6], [0, 1, 2, 3, 4], [7], 3, [5, 9, 7, 6], [1, 2, 3, 4]),
        ([7, 5, 9, 7, 6], [0, 1, 2, 3, 4], [5], 3, [7, 9, 7, 6], [0, 2, 3, 4]),
        ([7, 5, 9, 7, 6], [0, 1, 2, 3, 4], [], 3, [7, 5, 9, 7, 6], [0, 1, 2, 3, 4]),
        ([7, 5, 7, 7, 6], [0, 1, 2, 3, 4], [7, 5, 1], 3, [7, 7, 6], [2, 3, 4]),
        ([1, 2, 3, 4, 5], [0, 1, 2, 3, 4], [1, 4], 3, [2, 3, 4, 5], [1, 2, 3, 4]),
    ],
)
def test_manage_boundary_common_tokens(
    segm,
    t_st,
    trusted_list,
    lookahead,
    exp_segm,
    exp_time,
):
    ret_segm, ret_time = manage_boundary_common_tokens(
        segm, t_st, trusted_list, lookahead
    )
    assert exp_segm == ret_segm
    assert exp_time == ret_time


@pytest.mark.parametrize(
    "ex_input, expected",
    [
        ([[0, 1], [3], [6, 7, 9, 9]], [[0, 1, 3, 6, 7, 9, 9]]),
        ([[0, 1], [], [6, 7, 9, 9]], [[0, 1, 6, 7, 9, 9]]),
        ([[0, 1], [6, 7, 9, 9]], [[0, 1, 6, 7, 9, 9]]),
    ],
)
def test_combine_predictions(ex_input, expected):
    returned = combine_predictions(ex_input)
    assert expected == returned
