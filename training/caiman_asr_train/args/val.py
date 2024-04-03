import os
import time
from argparse import ArgumentParser, Namespace

from beartype import beartype

from caiman_asr_train.args.hugging_face import add_basic_hugging_face_args
from caiman_asr_train.args.shared import add_shared_args
from caiman_asr_train.train_utils.distributed import print_once


def val_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="RNN-T Training Reference")

    training = parser.add_argument_group("training/validation setup")
    training.add_argument(
        "--no_cudnn_benchmark",
        action="store_false",
        default=True,
        help="Disable cudnn benchmark",
    )
    training.add_argument("--seed", default=1, type=int, help="Random seed")
    training.add_argument(
        "--local_rank",
        default=os.getenv("LOCAL_RANK", 0),
        type=int,
        help="GPU id used for distributed processing",
    )

    optim = parser.add_argument_group("optimization setup")
    optim.add_argument(
        "--val_batch_size", default=1, type=int, help="Evaluation time batch size"
    )

    io = parser.add_argument_group("feature and checkpointing setup")
    io.add_argument(
        "--dali_device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Use DALI pipeline for fast data processing",
    )
    io.add_argument(
        "--ckpt",
        "--checkpoint",
        default="/results/RNN-T_best_checkpoint.pt",
        type=str,
        help="Path to the checkpoint to use",
    )
    io.add_argument(
        "--model_config",
        default="/workspace/training/configs/testing-1023sp_run.yaml",
        type=str,
        help="Path of the model configuration file",
    )
    io.add_argument(
        "--val_manifests",
        type=str,
        required=False,
        default=["/datasets/LibriSpeech/librispeech-dev-clean-wav.json"],
        nargs="+",
        help="Paths of the evaluation datasets manifest files. "
        "Ignored if --read_from_tar=True",
    )
    io.add_argument(
        "--read_from_tar",
        action="store_true",
        default=False,
        help="Read data from tar files instead of json manifest files",
    )
    io.add_argument(
        "--val_tar_files",
        type=str,
        required=False,
        default=None,
        nargs="+",
        help="Paths of the evaluation dataset tar files. "
        "Ignored if --read_from_tar=False.",
    )
    io.add_argument(
        "--dataset_dir",
        "--data_dir",
        type=str,
        help="Root dir of dataset",
        default="/datasets/LibriSpeech",
    )
    io.add_argument(
        "--output_dir",
        type=str,
        default="/results",
        help="Directory for logs and checkpoints",
    )
    io.add_argument(
        "--log_file", type=str, default=None, help="Path to save the logfile."
    )
    io.add_argument(
        "--max_symbol_per_sample",
        type=int,
        default=None,
        help="""Maximum number of symbols that can be decoded per sample during validation.
        By default None, which can lead to different WERs than reported during training;
        see `--max_symbol_per_sample` in `args/train.py`.""",
    )
    io.add_argument(
        "--calculate_loss",
        action="store_true",
        help="""Calculate transducer loss during validation; increases GPU
        memory usage and not supported with `--cpu` flag.""",
    )
    io.add_argument(
        "--timestamp",
        default=time.strftime("%Y_%m_%d_%H_%M_%S"),
        type=str,
        help="Timestamp to use for logging",
    )
    io.add_argument(
        "--nth_batch_only",
        default=None,
        type=int,
        help="Only evaluate the nth batch, useful for debugging",
    )
    io.add_argument(
        "--num_gpus",
        default=1,
        type=int,
        help="""Number of GPUs to use for validation. There are num_gpus processes,
        each running a copy of val.py on one GPU.""",
    )
    io.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="""Run validation on the CPU. It supports additional functionality
        """,
    )
    add_basic_hugging_face_args(parser)
    add_shared_args(parser)
    return parser


@beartype
def check_val_arguments(args: Namespace) -> None:
    """Check that the validation arguments are valid."""
    if args.cpu:
        print_once("Running on CPU")
        assert (
            not args.calculate_loss
        ), "Loss function calculation is not supported on CPU"

    if args.num_gpus > 1:
        assert not args.cpu, "Cannot use valCPU with multiple processes"
