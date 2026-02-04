from argparse import Namespace

import pytest

from caiman_asr_train.utils import frame_width
from caiman_asr_train.utils.frame_width import (
    encoder_output_frame_width,
    initial_padding,
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


def test_initial_padding_turn_off(monkeypatch):
    args = Namespace(turn_off_initial_padding=True, model_config="dummy")
    assert initial_padding(args) == 0.0


@pytest.mark.parametrize(
    "window_size,window_stride,expected",
    [
        (0.025, 0.010, 0.015),
        (0.020, 0.020, 0.0),
        (0.5, 0.1, 0.4),
    ],
)
def test_initial_padding_reads_config(
    monkeypatch, window_size, window_stride, expected
):
    args = Namespace(turn_off_initial_padding=False, model_config="dummy")

    config_dict = {
        "input_train": {
            "filterbank_features": {
                "window_size": window_size,
                "window_stride": window_stride,
            }
        },
        "input_val": {
            "filterbank_features": {
                "window_size": window_size,
                "window_stride": window_stride,
            }
        },
    }

    monkeypatch.setattr(frame_width.config, "load", lambda _: config_dict)

    assert expected == pytest.approx(initial_padding(args))
