import argparse
import os
import time
from argparse import Namespace
from pathlib import Path

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
from caiman_asr_train.lhotse.config_build import validate_dataset_yaml


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
    training.add_argument(
        "--eoi_drop_fraction",
        type=float,
        default=0.5,
        help="Fraction of utterances to drop EOI from, ignored if not appending EOI",
    )
    training.add_argument(
        "--inspect_memory_usage",
        type=str,
        help="Append memory usage statistics to the specified file path",
    )

    sampler = parser.add_argument_group("Sampler setup")

    sampler.add_argument(
        "--sampling_mode",
        type=str,
        default="1D-bucketing-dyn",
        choices=[
            "duration",
            "fixed",
            "1D-bucketing",
            "1D-bucketing-dyn",
            "2D-bucketing",
            "2D-bucketing-fixed",
            "2D-bucketing-singlet",
        ],
    )

    sampler.add_argument(
        "--singlet_bucket",
        type=int,
        help="For 2D-bucketing-singlet mode, the index of the single bucket to use",
    )

    sampler.add_argument(
        "--buckets",
        type=str,
        help="Path to precomputed buckets file",
    )

    # Fixed batching
    sampler.add_argument(
        "--batch_size",
        type=int,
        help="Fixed batch-size for fixed sampling and singlet 2D-bucketing modes",
    )

    # Variables for dynamic batching
    sampler.add_argument(
        "--batch_duration",
        type=float,
        help="Batch duration for dynamic batching",
    )
    sampler.add_argument(
        "--batch_tokens",
        type=int,
        help="Batch size in tokens",
    )
    sampler.add_argument(
        "--batch_area",
        type=float,
        help="Batch size in area (tokens * duration)",
    )
    sampler.add_argument(
        "--batch_vram",
        type=float,
        help="Batch size in vram (MB) for area-batching",
    )

    sampler.add_argument(
        "--ols_vram_coeff",
        type=float,
        nargs=3,
        help="OLS regression coefficients for area to VRAM estimation. "
        "Provide three coefficients in the order: [coeff_time, coeff_toks, coeff_area].",
    )

    sampler.add_argument(
        "--num_dur_buckets", type=int, default=10, help="Number of duration buckets"
    )
    sampler.add_argument(
        "--num_tok_buckets", type=int, default=4, help="Number of token buckets"
    )
    sampler.add_argument(
        "--optimize_buckets",
        action="store_true",
        help="Compute optimal 1D/2D bucketing boundaries, these bucket boundaries will"
        "be written to `out_dir/bins*`, they will not overwrite the existing bucketing "
        "settings used for training (if any). The train will terminate after computing "
        "the optimal buckets. To use the optimal buckets at train time pass the path "
        "to the --buckets flag on the next invocation of training.",
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
        "--grad_accumulation_batches",
        default=8,
        type=int,
        help="Number of batches that must be accumulated for a single model update (step)",
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
        default=6.0,
        type=float,
        help="log2 of the minimum value of the AMP gradient scaler. The default "
        "floor of 2^6 is far below any healthy scale (~2^10-2^16) so it only "
        "engages during an instability, where it stops the scale free-falling "
        "into the fp16-underflow regime that blocks recovery. Set very negative "
        "(e.g. -100) to effectively disable",
    )
    optim.add_argument(
        "--grad_norm_skip_factor",
        default=20.0,
        type=float,
        help="Skip the optimizer step when the unscaled gradient norm exceeds this "
        "factor times its running average. Guards against finite-but-huge "
        "gradients that pass the AMP inf check during an instability cascade. "
        "Set to 0 to disable the check",
    )
    optim.add_argument(
        "--skip_decay_momentum",
        default=0.99,
        type=float,
        help="Momentum for the decoupled weight decay applied on skipped optimizer "
        "steps. On each consecutive skip a velocity accumulates (v <- momentum*v + 1) "
        "and scales the lr*weight_decay decay, so the longer a skip episode runs the "
        "harder the (growing) weights are decayed; it resets on the next real step. "
        "0 recovers the un-augmented fixed decay, 1 ramps linearly, and 0<m<1 "
        "saturates at 1/(1-m).",
    )

    io = parser.add_argument_group("feature and checkpointing setup")
    io.add_argument(
        "--loader_train_device",
        type=str,
        choices=["cpu", "gpu"],
        default="gpu",
        help="Set the device for the training data loader",
    )
    io.add_argument(
        "--loader_val_device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Set the device for the validation data loader",
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
        "--val_nth_batch_only",
        default=None,
        type=int,
        help="Only evaluate the nth val batch during training, useful for debugging.",
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
        type=str,
        help="Path of the model configuration file",
        required=True,
    )
    io.add_argument(
        "--train_dataset_yaml",
        type=str,
        required=True,
        help="Path to training dataset config (.yaml) file",
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
    io.add_argument(
        "--skip_first_val_eval",
        action="store_true",
        help="Skip the first validation evaluation after training starts.",
    )
    io.add_argument(
        "--skip_final_val_eval",
        action="store_true",
        help="Skip the final validation evaluation at the end of training.",
    )
    io.add_argument(
        "--conditional",
        action="store_true",
        help="Enable conditional PnC/NoPnC training.",
    )
    io.add_argument(
        "--conditional_language",
        action="store_true",
        help="Enable conditional language training (requires --conditional).",
    )
    io.add_argument(
        "--pnc_prob",
        type=float,
        default=0.25,
        help="Probability that a training utterance is PNC",
    )
    io.add_argument(
        "--lang_mask_prob",
        type=float,
        default=None,
        help="""Probability of masking the lang_tok with
          <lang_detect> in a training utterance""",
    )
    io.add_argument(
        "--cuts_dump_path",
        type=str,
        default=None,
        help="""Path to dump the cuts used for training. This is useful for debugging as
        it contains the pre- and post-normalization cuts used for training.
        The path should be in the format /results/<filename>.jsonl. Each process
        (rank) writes to its own file, named <filename>_rank<N>.jsonl, so make sure
        to use a small number of training steps.""",
    )

    add_noise_augmentation_args(parser)
    add_shared_args(parser)
    add_delay_penalty_args(parser)
    add_star_args(parser)
    add_eos_train_args(parser)
    return parser


def verify_train_args(args: Namespace) -> Namespace:
    # check data path args

    validate_dataset_yaml(args.train_dataset_yaml)

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

    if args.lang_mask_prob is not None:
        if not args.conditional:
            raise ValueError("language masking requires conditional mode.")
        if not args.conditional_language:
            raise ValueError("language masking requires conditional_language mode.")
        if args.lang_mask_prob < 0 or args.lang_mask_prob > 1:
            raise ValueError("args.lang_mask_prob must be between 0 and 1")

    return args
