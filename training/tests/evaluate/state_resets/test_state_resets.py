import pytest
import torch

from caiman_asr_train.evaluate.state_resets import (
    extend_feats_with_overlaps,
    get_segmenting_info,
    get_state_resets_feats,
    get_state_resets_stats,
    validate_state_resets_arguments,
)


@pytest.mark.parametrize(
    "feats, over_frames, segm_frames, n_seg, exp_feats",
    [
        (
            torch.tensor([[[0]], [[1]], [[2]], [[3]]]),
            2,
            3,
            2,
            torch.tensor([[[0]], [[1]], [[2]], [[1]], [[2]], [[3]]]),
        ),
        (
            torch.tensor([[[0]], [[1]], [[2]], [[3]], [[4]]]),
            3,
            4,
            2,
            torch.tensor([[[0]], [[1]], [[2]], [[3]], [[1]], [[2]], [[3]], [[4]]]),
        ),
        (
            torch.tensor([[[0]], [[1]], [[2]], [[3]], [[4]], [[5]]]),
            2,
            4,
            2,
            torch.tensor([[[0]], [[1]], [[2]], [[3]], [[2]], [[3]], [[4]], [[5]]]),
        ),
        (
            torch.tensor([[[0]], [[1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]]]),
            2,
            4,
            3,
            torch.tensor(
                [
                    [[0]],
                    [[1]],
                    [[2]],
                    [[3]],
                    [[2]],
                    [[3]],
                    [[4]],
                    [[5]],
                    [[4]],
                    [[5]],
                    [[6]],
                    [[7]],
                ]
            ),
        ),
        (
            torch.tensor([[[0]], [[1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]]]),
            6,
            7,
            2,
            torch.tensor(
                [
                    [[0]],
                    [[1]],
                    [[2]],
                    [[3]],
                    [[4]],
                    [[5]],
                    [[6]],
                    [[1]],
                    [[2]],
                    [[3]],
                    [[4]],
                    [[5]],
                    [[6]],
                    [[7]],
                ]
            ),
        ),
    ],
)
def test_extend_feats_with_overlaps(feats, over_frames, segm_frames, n_seg, exp_feats):
    returned_feats = extend_feats_with_overlaps(
        feats.float(), over_frames, segm_frames, n_seg
    )
    assert torch.allclose(returned_feats, exp_feats.float())


@pytest.mark.parametrize(
    "feats, feat_lens, segm_frames, overl_frames, exp_feats, exp_feat_lens",
    [
        (  # short utterance - no state resets is applied
            torch.tensor([[0], [1], [2], [3], [4], [5]]),
            torch.tensor([6]),
            10,
            5,
            torch.tensor([[0], [1], [2], [3], [4], [5]]),
            torch.tensor([6]),
        ),
        (
            torch.tensor([[[0]], [[1]], [[2]], [[3]]]),
            torch.tensor([4]),
            3,
            2,
            torch.tensor([[[0], [1]], [[1], [2]], [[2], [3]]]),
            torch.tensor([3, 3]),
        ),
        (
            torch.tensor([[[0]], [[1]], [[2]], [[3]], [[4]]]),
            torch.tensor([5]),
            3,
            2,
            torch.tensor([[[0], [1], [2]], [[1], [2], [3]], [[2], [3], [4]]]),
            torch.tensor([3, 3, 3]),
        ),
        (
            torch.tensor(
                [
                    [[0]],
                    [[1]],
                    [[2]],
                    [[3]],
                    [[4]],
                    [[5]],
                    [[6]],
                    [[7]],
                    [[8]],
                    [[9]],
                    [[10]],
                    [[11]],
                ]
            ),
            torch.tensor([12]),
            7,
            2,
            torch.tensor(
                [
                    [[0], [5]],
                    [[1], [6]],
                    [[2], [7]],
                    [[3], [8]],
                    [[4], [9]],
                    [[5], [10]],
                    [[6], [11]],
                ]
            ),
            torch.tensor([7, 7]),
        ),
    ],
)
def test_get_state_resets_feats(
    feats, feat_lens, segm_frames, overl_frames, exp_feats, exp_feat_lens
):
    returned_feats, returned_feat_lens = get_state_resets_feats(
        feats.float(), feat_lens, segm_frames, overl_frames
    )
    assert torch.allclose(returned_feats, exp_feats.float())
    assert torch.equal(returned_feat_lens, exp_feat_lens)


@pytest.mark.parametrize(
    "segment, overlap", [(-15.0, 0.0), (15.0, 25.0), (0.0, 1.0), (1.0, 1.0)]
)
def test_validate_state_resets_arguments(segment, overlap):
    with pytest.raises(ValueError):
        validate_state_resets_arguments(segment, overlap)


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            (
                16.0,
                2.0,
                {
                    "input_train": {
                        "filterbank_features": {"window_stride": 1.0},
                        "frame_splicing": {"frame_stacking": 1, "frame_subsampling": 1},
                    }
                },
            ),
            (16, 2),
        ),
        (
            (
                16.0,
                2.0,
                {
                    "input_train": {
                        "filterbank_features": {"window_stride": 0.01},
                        "frame_splicing": {"frame_stacking": 3, "frame_subsampling": 3},
                    }
                },
            ),
            (533, 67),
        ),
        (
            (
                15.0,
                0.0,
                {
                    "input_train": {
                        "filterbank_features": {"window_stride": 0.01},
                        "frame_splicing": {"frame_stacking": 3, "frame_subsampling": 3},
                    }
                },
            ),
            (500, 0),
        ),
    ],
)
def test_get_state_resets_stats(test_input, expected):
    returned_segments, returned_overlaps = get_state_resets_stats(
        test_input[0], test_input[1], test_input[2]
    )
    assert returned_segments == expected[0]
    assert returned_overlaps == expected[1]


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ((36569, 67, 533), (312, 79)),
        ((11, 1, 4), (2, 4)),
        ((13, 1, 4), (0, 4)),
        ((13, 1, 3), (0, 6)),
        ((13, 2, 5), (1, 4)),
        ((13, 2, 6), (1, 3)),
        ((13, 2, 7), (4, 3)),
        ((12, 2, 7), (0, 2)),
    ],
)
def test_get_segmenting_info(test_input, expected):
    n_chunks, padding = get_segmenting_info(test_input[0], test_input[1], test_input[2])
    assert padding == expected[0]
    assert n_chunks == expected[1]
