import pytest

from caiman_asr_train.utils.frame_width import (
    encoder_output_frame_width,
    input_feat_frame_width,
)


def test_encoder_frame_width(mini_model_factory):
    _, config_path = mini_model_factory()
    # Note this is only true for the mini model
    assert encoder_output_frame_width(config_path) == 0.03


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            {
                "input_train": {
                    "filterbank_features": {"window_stride": 1.0},
                    "frame_splicing": {"frame_stacking": 1, "frame_subsampling": 1},
                },
            },
            1.0,
        ),
        (
            {
                "input_train": {
                    "filterbank_features": {"window_stride": 0.0},
                    "frame_splicing": {"frame_stacking": 1, "frame_subsampling": 1},
                },
            },
            0.0,
        ),
    ],
)
def test_input_feat_frame_width(test_input, expected):
    returned = input_feat_frame_width(test_input)
    assert returned == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            {
                "rnnt": {"enc_stack_time_factor": 2},
                "input_train": {
                    "filterbank_features": {"window_stride": 1.0},
                    "frame_splicing": {"frame_stacking": 1, "frame_subsampling": 3},
                },
            },
            2.0,
        ),
    ],
)
def test_raises_input_feat_frame_width(test_input, expected):
    with pytest.raises(AssertionError):
        _ = input_feat_frame_width(test_input)
