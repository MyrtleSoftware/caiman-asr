import os
import time
from argparse import ArgumentParser, Namespace

from beartype import beartype

from caiman_asr_train.args.argparser import MyrtleArgumentParser
from caiman_asr_train.args.shared import add_shared_args
from caiman_asr_train.train_utils.distributed import print_once


@beartype
def val_arg_parser() -> ArgumentParser:
    parser = MyrtleArgumentParser(description="RNN-T Training Reference")

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

    mutex = optim.add_mutually_exclusive_group()
    mutex.add_argument("--val_batch_size", type=int, help="Evaluation time batch size")
    mutex.add_argument(
        "--val_batch_duration",
        type=float,
        help="Eval time batch duration",
        default=None,
    )

    io = parser.add_argument_group("feature and checkpointing setup")
    io.add_argument(
        "--loader_val_device",
        type=str,
        choices=["cpu", "gpu"],
        default="gpu",
        help="Set the validation data loader device.",
    )
    io.add_argument(
        "--ckpt",
        "--checkpoint",
        default="/results/RNN-T_best_checkpoint.pt",
        type=str,
        help="""Path to the checkpoint to use, or directory containing
        checkpoints to validate all .pt files within.""",
    )
    io.add_argument(
        "--model_config",
        type=str,
        help="Path of the model configuration file",
        required=True,
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
    io.add_argument(
        "--uuid",
        type=str,
        default="0",
        help="Unique ID, propagated into validation results.",
    )
    io.add_argument(
        "--feature_dump_dir",
        type=str,
        default=None,
        help="Directory to dump the features to",
    )

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

    if args.val_batch_duration is None and args.val_batch_size is None:
        args.val_batch_duration = 14_400.0
