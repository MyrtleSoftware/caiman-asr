#! /usr/bin/env python3

from argparse import Namespace

from beartype import beartype

from caiman_asr_train.rnnt import config


@beartype
def initial_padding(args: Namespace) -> float:
    if args.turn_off_initial_padding:
        return 0.0

    conf = config.load(args.model_config)

    window_size = conf["input_train"]["filterbank_features"]["window_size"]
    window_stride = conf["input_train"]["filterbank_features"]["window_stride"]

    assert window_size == conf["input_val"]["filterbank_features"]["window_size"]
    assert window_stride == conf["input_val"]["filterbank_features"]["window_stride"]

    return window_size - window_stride


def encoder_output_frame_width(model_config_path: str) -> float:
    """Calculates the effective frame width in seconds from model config.

    This function will return the duration in seconds of a frame.
    It takes under consideration the stacking that is happening in the encoder.

    Parameters
    ----------
    model_config_path
        model configuration filepath

    Returns
    -------
    float
        the window size duration in seconds
    """
    model_config = config.load(model_config_path)
    stack_factor = model_config["rnnt"]["enc_stack_time_factor"]
    frame_width = input_feat_frame_width(model_config)

    return stack_factor * frame_width


@beartype
def input_feat_frame_width(model_config: dict) -> float:
    """Calculates the frame width (before the stacking) in seconds from model config.

    This function will return the duration in seconds of a frame width. This is before
    any time stacking happens in the encoder.

    Parameters
    ----------
    model_config
        model configuration file

    Returns
    -------
    float
        the frame duration in seconds before time stacking
    """
    window_stride = model_config["input_train"]["filterbank_features"]["window_stride"]
    frame_stacking = model_config["input_train"]["frame_splicing"]["frame_stacking"]
    frame_subsampling = model_config["input_train"]["frame_splicing"][
        "frame_subsampling"
    ]

    assert (
        frame_stacking == frame_subsampling
    ), "ERROR: please use the same frame stacking and frame subsampling."

    return window_stride * frame_stacking
