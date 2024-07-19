#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
from pathlib import Path

from beartype import beartype
from beartype.typing import Optional

from caiman_asr_train.args.decoder import add_decoder_args
from caiman_asr_train.args.hugging_face import add_basic_hugging_face_args
from caiman_asr_train.args.mel_feat_norm import (
    add_mel_feat_norm_args,
    check_mel_feat_norm_args,
)
from caiman_asr_train.train_utils.distributed import print_once


@beartype
def add_shared_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--turn_off_initial_padding",
        action="store_true",
        help="""By default, audio is pre-padded with (window_size - window_stride)
        seconds of silence to match the asr-server's behaviour. This is 10ms for the
        testing model, and 15ms for the base/large models. This padding does not
        affect WER. This option turns that off.""",
    )
    parser.add_argument(
        "--prob_val_narrowband",
        type=float,
        default=0.0,
        help="Probability that a batch of validation audio gets downsampled to 8kHz"
        " and then upsampled to original sample rate",
    )
    parser.add_argument(
        "--inspect_audio",
        action="store_true",
        help="""Save audios (after augmentations are applied) to /results/augmented_audios.
        This will slow down DALI""",
    )
    parser.add_argument(
        "--n_utterances_only",
        default=None,
        type=int,
        help="Abridge the dataset to only this many utterances, selected randomly",
    )
    parser.add_argument(
        "--skip_init",
        action="store_true",
        default=False,
        help="If true do not re-initialise things that should only be initialised once",
    )
    parser.add_argument(
        "--called_by_torchrun",
        action="store_true",
        help="""When a multiGPU script is split into multiple processes, this flag is
        set so that the processes don't try to split again. The user should not set this
        flag""",
    )
    parser.add_argument(
        "--max_inputs_per_batch",
        default=int(1e7),
        type=int,
        help="During decoding, the encoder will try to reduce the batch size to keep "
        "torch.numel(audio_features) below this value, to lower GPU memory usage. "
        "Note this default is for an 11GB GPU",
    )
    parser.add_argument(
        "--val_from_dir",
        action="store_true",
        default=False,
        help="Read data from directories",
    )
    parser.add_argument(
        "--val_audio_dir",
        type=str,
        required=False,
        default=None,
        help="Path to the directory containing the audio files.",
    )
    parser.add_argument(
        "--val_txt_dir",
        type=str,
        required=False,
        default=None,
        help="Path to the directory containing the transcripts of the audio files. "
        "It can be the same as the audio directory.",
    )
    parser.add_argument(
        "--dump_preds",
        action="store_true",
        default=False,
        help="Dump text predictions to /{output_dir}/preds.txt",
    )
    parser.add_argument(
        "--dali_processes_per_cpu",
        type=float,
        default=1.0,
        help="Number of DALI processes per CPU thread. This automatically adjusts for "
        "multi-GPU training. The default of 1 is optimal for an 8xA100 (80 GB) server. "
        "If you are running two different trains on the same machine, with the GPUs "
        "partitioned between them, setting this to 0.5 may speed up DALI",
    )
    parser.add_argument(
        "--profiler",
        action="store_true",
        help="""Enable profiling with yappi and save top/nvidia-smi logs.
        This may slow down training/validation""",
    )
    add_state_reset_args(parser)
    add_mel_feat_norm_args(parser)
    add_basic_hugging_face_args(parser)
    add_wer_analysis_args(parser)
    add_decoder_args(parser)


@beartype
def add_wer_analysis_args(parser: ArgumentParser) -> None:
    wer_args = parser.add_argument_group("WER analysis")
    wer_args.add_argument(
        "--breakdown_wer",
        action="store_true",
        help="""Print an analysis of how much casing and punctuation affect WER.
        Most useful when the reference text contains casing and punctuation""",
    )
    default_punctuation = ",.?!;:-"
    wer_args.add_argument(
        "--breakdown_chars",
        type=str,
        default=default_punctuation,
        help=f"Which characters to analyze. Default is '{default_punctuation}'",
    )


@beartype
def add_state_reset_args(parser: ArgumentParser) -> None:
    sr_args = parser.add_argument_group("State Reset setup")
    sr_args.add_argument(
        "--sr_segment",
        type=float,
        default=15.0,
        help="State resets segment duration in seconds.",
    )
    sr_args.add_argument(
        "--sr_overlap",
        type=float,
        default=3.0,
        help="State resets overlapping duration in seconds. Used only when segment "
        "duration is set.",
    )


@beartype
def check_state_reset_args(args: Namespace) -> None:
    """require validation batch size = 1 if state resets is used"""
    if args.sr_segment:
        assert (
            args.val_batch_size == 1
        ), "Please set --val_batch_size=1 to use state resets."
        print_once(
            f"If running validation: state resets are used every {args.sr_segment} "
            f"seconds with overlap of {args.sr_overlap} seconds."
        )


@beartype
def check_shared_args(args: Namespace) -> None:
    if args.val_from_dir:
        assert validation_directories_provided(
            audio_dir=args.val_audio_dir, txt_dir=args.val_txt_dir
        ), (
            f"Argument {args.val_from_dir=} is set, {args.val_audio_dir=} "
            f"and {args.val_txt_dir=} must be provided"
        )
        check_directories_are_valid(
            args.val_audio_dir, args.val_txt_dir, args.dataset_dir
        )
        print_once(
            f"Running validation from directories {args.val_audio_dir=} "
            f"and {args.val_txt_dir=}. The {args.val_manifests=} argument will be ignored."
        )

    if args.read_from_tar or args.use_hugging_face:
        assert (
            args.n_utterances_only is None
        ), "n_utterances_only is not supported when reading from tar files or hugging face"
    else:
        assert (
            args.val_manifests is not None
        ), f"Must provide {args.val_manifests=} if not reading from tar files"
        "or not reading audio and transcripts from directories"

    check_mel_feat_norm_args(args)
    check_state_reset_args(args)


@beartype
def validation_directories_provided(
    audio_dir: Optional[str], txt_dir: Optional[str]
) -> bool:
    return audio_dir is not None and txt_dir is not None


@beartype
def check_directories_are_valid(audio_dir: str, txt_dir: str, dataset_dir) -> None:
    """
    Check that the provided directories exist, and contain the same file names.

    Parameters
    ----------
    audio_dir
        directory containing audio files
    txt_dir
        directory containing text files
    dataset_dir
        directory containing audio_dir and txt_dir

    Returns
    -------
    None

    Raises
    ------
    ValueError:
        If either of the directories do not exist, or if they do not contain the same
        file names
    """
    data_path = Path(dataset_dir)
    audio_abs_path = data_path.joinpath(audio_dir)
    txt_abs_path = data_path.joinpath(txt_dir)

    # check directories exist
    if not audio_abs_path.is_dir():
        raise ValueError(f"{audio_dir=} is not a directory")
    if not txt_abs_path.is_dir():
        raise ValueError(f"{txt_dir=} is not a directory")

    # check directories contain the same file names
    audio_files = set(
        f.stem
        for f in audio_abs_path.glob("**/*")
        if f.is_file() and f.suffix in [".wav", ".flac"]
    )
    txt_files = set(
        f.stem
        for f in txt_abs_path.glob("**/*")
        if f.is_file() and f.suffix in [".txt"]
    )
    if audio_files != txt_files:
        raise ValueError(
            f"Audio and txt directories do not contain the same files. "
            f"Provided directories are: {audio_dir=}, txt files: {txt_dir=}"
        )
