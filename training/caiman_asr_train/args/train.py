import argparse
import os
import time
from argparse import Namespace
from pathlib import Path

import yaml
from beartype import beartype

from caiman_asr_train.args.argparser import MyrtleArgumentParser
from caiman_asr_train.args.delay_penalty import (
    add_delay_penalty_args,
    verify_delay_penalty_args,
)
from caiman_asr_train.args.eos import add_eos_train_args
from caiman_asr_train.args.noise_augmentation import add_noise_augmentation_args
from caiman_asr_train.args.shared import add_shared_args
from caiman_asr_train.args.star import add_star_args
from caiman_asr_train.data.schema import DatasetSchemaValidator


@beartype
def train_arg_parser() -> argparse.ArgumentParser:
    parser = MyrtleArgumentParser(description="RNN-T Training Reference")

    training = parser.add_argument_group("training setup")
    training.add_argument(
        "--training_steps",
        default=100000,
        type=int,
        help="Number of steps for the entire training",
    )
    training.add_argument(
        "--warmup_steps",
        default=1632,
        type=int,
        help="Initial steps of increasing learning rate",
    )
    training.add_argument(
        "--hold_steps",
        default=18000,
        type=int,
        help="Constant max learning rate steps after warmup",
    )
    training.add_argument(
        "--half_life_steps",
        default=10880,
        type=int,
        help="half life (in steps) for exponential learning rate decay",
    )
    training.add_argument(
        "--no_cudnn_benchmark",
        action="store_true",
        default=False,
        help="Disable cudnn benchmark",
    )
    training.add_argument(
        "--no_amp",
        action="store_true",
        default=False,
        help="Turn off pytorch mixed precision training. This will slow down training",
    )
    training.add_argument("--seed", default=1, type=int, help="Random seed")
    training.add_argument(
        "--local_rank",
        default=os.getenv("LOCAL_RANK", 0),
        type=int,
        help="GPU id used for distributed training",
    )
    training.add_argument(
        "--weights_init_scale",
        default=0.5,
        type=float,
        help="If set, overwrites value in config.",
    )
    training.add_argument(
        "--hidden_hidden_bias_scale",
        "--hidden_hidden_bias_scaled",
        type=float,
        help="If set, overwrites value in config.",
    )
    training.add_argument(
        "--die_if_wer_bad",
        action="store_true",
        help="If dev WER > 99% at step 10k or later, raise an error",
    )
    optim = parser.add_argument_group("optimization setup")
    optim.add_argument(
        "--num_gpus",
        default=8,
        type=int,
        help="""Number of GPUs to use for training. There are num_gpus processes,
        each running a copy of train.py on one GPU.""",
    )
    optim.add_argument(
        "--global_batch_size",
        default=1024,
        type=int,
        help="Effective batch size across all GPUs after grad accumulation",
    )
    optim.add_argument(
        "--grad_accumulation_batches",
        default=8,
        type=int,
        help="Number of batches that must be accumulated for a single model update (step)",
    )
    optim.add_argument(
        "--batch_split_factor",
        default=1,
        type=int,
        help="Multiple >=1 describing how much larger the encoder/prediction batch size "
        "is than the joint/loss batch size",
    )
    optim.add_argument(
        "--val_batch_size", default=16, type=int, help="Evaluation time batch size"
    )
    optim.add_argument(
        "--lr", "--learning_rate", default=4e-3, type=float, help="Peak learning rate"
    )
    optim.add_argument(
        "--min_lr",
        "--min_learning_rate",
        default=4e-4,
        type=float,
        help="minimum learning rate",
    )
    optim.add_argument(
        "--weight_decay",
        default=1e-2,
        type=float,
        help="Weight decay for the optimizer",
    )
    optim.add_argument(
        "--clip_norm",
        default=1,
        type=float,
        help="If provided, gradients will be clipped above this norm",
    )
    optim.add_argument("--beta1", default=0.9, type=float, help="Beta 1 for optimizer")
    optim.add_argument(
        "--beta2", default=0.999, type=float, help="Beta 2 for optimizer"
    )
    optim.add_argument(
        "--ema",
        type=float,
        default=0.999,
        help="Discount factor for exp averaging of model weights",
    )
    optim.add_argument(
        "-scaler_lbl2",
        "--grad_scaler_lower_bound_log2",
        default=None,
        type=float,
        help="The minimum value of the gradient scaler, set to None for no minimum",
    )

    io = parser.add_argument_group("feature and checkpointing setup")
    io.add_argument(
        "--dali_train_device",
        type=str,
        choices=["cpu", "gpu"],
        default="gpu",
        help="Use DALI pipeline for fast data processing",
    )
    io.add_argument(
        "--dali_val_device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Use DALI pipeline for fast data processing",
    )
    io.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the specified checkpoint or the last saved checkpoint.",
    )
    io.add_argument(
        "--fine_tune",
        action="store_true",
        help="Start training anew from the specified checkpoint.",
    )
    io.add_argument(
        "--ckpt", "--checkpoint", default=None, type=str, help="Path to a checkpoint"
    )
    io.add_argument(
        "--dont_save_at_the_end",
        action="store_true",
        help="Don't save model checkpoint at the end of training",
    )
    io.add_argument(
        "--save_frequency",
        default=5000,
        type=int,
        help=(
            "Checkpoint saving frequency in steps. If 0 (or None), only possibly save "
            "best and last checkpoints "
        ),
    )
    io.add_argument(
        "--val_frequency",
        default=1000,
        type=int,
        help="Number of steps between evaluations on dev set",
    )
    io.add_argument(
        "--log_frequency",
        default=1,
        type=int,
        help="Number of steps between printing training stats",
    )
    io.add_argument(
        "--prediction_frequency",
        default=1000,
        type=int,
        help="Number of steps between printing sample decodings",
    )
    io.add_argument(
        "--model_config",
        default="/workspace/training/configs/testing-1023sp_run.yaml",
        type=str,
        help="Path of the model configuration file",
    )
    io.add_argument(
        "--num_buckets",
        type=int,
        default=6,
        help="The number of buckets for the Bucketing Sampler, "
        "according to which audio files are grouped by audio duration "
        "and shuffled within each bucket. Setting it to 0 will use "
        "the RandomSampler.",
    )
    io.add_argument(
        "--randomize_first_n_epochs",
        type=int,
        default=0,
        help="Completely randomize the first n epochs regardless of bucketing",
    )
    io.add_argument(
        "--train_manifests",
        type=str,
        required=False,
        nargs="+",
        help="Paths of the training dataset manifest file"
        "Ignored if --read_from_tar=True",
    )
    io.add_argument(
        "--train_dataset_yaml",
        type=str,
        required=False,
        help="Path to training dataset config (.yaml) file",
    )
    io.add_argument(
        "--train_manifest_ratios",
        "--train_manifests_ratios",
        type=float,
        required=False,
        nargs="+",
        help="Ratio to weight each training manifest file in each epoch by. "
        " 1 2 means the model will see utterances from the second manifest "
        "twice as often as utterances from the first, "
        "regardless of the lengths of the manifests",
    )
    io.add_argument(
        "--relative_train_manifest_ratios",
        type=float,
        required=False,
        nargs="+",
        help="Ratio to weight each training manifest file in each epoch by. "
        " 1 2 means the second manifest will be upweighted by a factor of 2, "
        " but the lengths of the manifests will be taken into account",
    )
    io.add_argument(
        "--canary_exponent",
        type=float,
        help="The exponent for the canary manifest weights. Canary weighting is "
        "on by default (with an exponent of 0.75) unless relative/absolute "
        "manifest ratios are set. To explicitly disable canary weighting set "
        "this flag to negative value.",
    )
    io.add_argument(
        "--val_manifests",
        type=str,
        required=False,
        default=["/datasets/LibriSpeech/librispeech-dev-clean-flac.json"],
        nargs="+",
        help="Paths of the evaluation datasets manifest files"
        "Ignored if --read_from_tar=True",
    )
    io.add_argument(
        "--read_from_tar",
        action="store_true",
        default=False,
        help="Read data from tar files instead of json manifest files",
    )
    io.add_argument(
        "--train_tar_files",
        type=str,
        required=False,
        default=None,
        nargs="+",
        help="One or more paths or globs for the training dataset tar files. "
        "Ignored if --read_from_tar=False. Must be provided if "
        "--read_from_tar=True.",
    )
    io.add_argument(
        "--val_tar_files",
        type=str,
        required=False,
        default=None,
        nargs="+",
        help="Paths (or globs) of the evaluation datasets tar files."
        "Ignored if --read_from_tar=False. Must be provided if "
        "--read_from_tar=True.",
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
        "--log_file", type=str, default=None, help="Path to save the training logfile."
    )
    io.add_argument(
        "--max_symbol_per_sample",
        type=int,
        default=300,
        help="""Maximum number of symbols that can be decoded per sample during training.
        By default, capped to 300 to prevent untrained model from decoding forever.""",
    )
    io.add_argument(
        "--rsp_seq_len_freq",
        type=int,
        nargs="+",
        default=[99, 0, 1],
        help="""Controls frequency and amount of random state passing

        [100,4,5] means that there will be 100 normal utterances, 4 utterances
        that are 2x longer, and 5 utterances that are 3x longer. Then training
        will loop back to 100 more normal utterances, etc

        This list can be longer: [99,0,0,0,1] means that there will be 99 normal
        utterances and 1 utterance that is 5x longer.

        The model does not always sees exactly 99 normal utterances followed by
        1 longer utterance, since the utterances are in random order.

        "1 utterance that is 5x longer" is implemented by passing the state
        through 5 consecutive utterances. Hence the 5x longer utterance and the
        99 normal utterances use up 104 normal utterances of training data

        To do no state passing, pass [1].

        Experiments suggest a default of [99,0,1]
        """,
    )
    io.add_argument(
        "--rsp_delay",
        type=int,
        default=None,
        help="""Steps of training to do before turning on random state passing. If this
        is None the value defaults to the one set by "
        "caiman_asr_train/train_utils/rsp.py::set_rsp_delay_default.
        See that docstring for more information.
        """,
    )
    io.add_argument(
        "--timestamp",
        default=time.strftime("%Y_%m_%d_%H_%M_%S"),
        type=str,
        help="Timestamp to use for logging",
    )
    io.add_argument(
        "--skip_state_dict_check",
        action="store_true",
        default=False,
        help="Disable checking of model architecture at start of training. This "
        "will result in trained models that are incompatible with downstream inference "
        "server and is intended for experimentation only.",
    )
    io.add_argument(
        "--prob_train_narrowband",
        type=float,
        default=0.0,
        help="Probability that a batch of training audio gets downsampled to 8kHz"
        " and then upsampled to original sample rate",
    )
    io.add_argument(
        "--skip_val_loss",
        action="store_true",
        help="""Only calculate WER, not loss, on validation set.
        Saves VRAM when validation set contains long utterances""",
    )
    io.add_argument(
        "--log_verbose_utterance_statistics",
        action="store_true",
        help="Perform expensive logging of utterance statistics",
    )

    add_noise_augmentation_args(parser)
    add_shared_args(parser)
    add_delay_penalty_args(parser)
    add_star_args(parser)
    add_eos_train_args(parser)
    return parser


def verify_train_args(args: Namespace) -> Namespace:
    # check data path args

    if args.train_dataset_yaml:
        assert (
            not args.train_manifests
        ), "Cannot provide both --train_dataset_yaml and --train_manifests."
        assert (
            not args.train_manifest_ratios
        ), "Cannot provide both --train_dataset_yaml and --train_manifest_ratios."
        assert (
            not args.relative_train_manifest_ratios
        ), "Cannot provide both --train_dataset_yaml and --relative_train_manifest_ratios."
        assert (
            not args.canary_exponent or args.canary_exponent < 0
        ), "Cannot provide both --train_dataset_yaml and --canary_exponent."

        # Load and validate the YAML file
        with open(args.train_dataset_yaml, "r") as f:
            dataset_config = yaml.safe_load(f)

        validator = DatasetSchemaValidator()
        validated_data = validator.validate(dataset_config)

        datasets = validated_data["datasets"]
        args.train_manifests = [datasets[name]["manifest"] for name in datasets]
        args.relative_train_manifest_ratios = [
            datasets[name]["weight"] for name in datasets
        ]

        assert len(args.train_manifests) > 0, "No valid datasets found in YAML."

    if not args.read_from_tar:
        assert (
            args.train_manifests is not None
        ), "Must provide train_manifests if not reading from tar"
        assert args.train_tar_files is None and args.val_tar_files is None, (
            "Must not provide tar files if not reading from tar but "
            f"{args.train_tar_files=} and {args.val_tar_files=}.\nDid you mean to "
            "pass --read_from_tar?"
        )

        assert (
            args.train_manifest_ratios is None
            or args.relative_train_manifest_ratios is None
        ), "Cannot set both --train_manifest_ratios and --relative_train_manifest_ratios"
        if args.train_manifest_ratios is not None:
            assert len(args.train_manifests) == len(
                args.train_manifest_ratios
            ), "Number of train manifests must match number of train manifest ratios"
        if args.relative_train_manifest_ratios is not None:
            assert len(args.train_manifests) == len(
                args.relative_train_manifest_ratios
            ), "Number of train manifests must match number of train manifest ratios"

        no_ratios = (
            args.train_manifest_ratios is None
            and args.relative_train_manifest_ratios is None
        )

        if args.canary_exponent is not None and args.canary_exponent < 0:
            # Explicit disable
            args.canary_exponent = None
        elif args.canary_exponent is None and no_ratios:
            # Default to canary weighting
            args.canary_exponent = 0.75

    else:
        assert (
            args.val_tar_files is not None
        ), "Must provide val_tar_files if --read_from_tar=True"
        assert (
            args.train_tar_files is not None
        ), "Must provide train_tar_files if --read_from_tar=True"
        assert (
            args.train_manifest_ratios is None
        ), "Manifest balancing only works with json manifests"
        assert (
            args.relative_train_manifest_ratios is None
        ), "Manifest balancing only works with json manifests"
        assert (
            args.canary_exponent is None
        ), "Canary weights not supported with tar files"

    mutual_exclusive_args = [
        args.canary_exponent is not None,
        args.train_manifest_ratios is not None,
        args.relative_train_manifest_ratios is not None,
    ]

    assert (
        mutual_exclusive_args.count(True) <= 1
    ), "Only one of the manifest ratio args should be set"

    if args.canary_exponent is not None:
        assert 0 < args.canary_exponent < 1, "Canary exponent must be between 0 and 1"

    verify_delay_penalty_args(args)

    out_dir = Path(args.output_dir)
    # fail if output dir already contains checkpoints
    if not args.resume and out_dir.exists() and any(out_dir.glob("*checkpoint*.pt")):
        error_msg = (
            f"{out_dir=} already contains checkpoints which would be overwritten by this "
            "command. Running training using the same output_dir as a previous command "
            "is only permitted when args.resume=True."
        )
        if args.fine_tune:
            error_msg += (
                " In the args.fine_tune=True case it is recommended to pass args.ckpt "
                "of the form /checkpoints/<ckpt_path> instead of /results/<ckpt_path> in "
                "order to avoid this error."
            )
        raise ValueError(error_msg)

    return args
