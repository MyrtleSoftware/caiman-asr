#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace

import torch
import yaml
from beartype import beartype

from caiman_asr_train.args.decoder import add_decoder_args
from caiman_asr_train.args.eos import add_eos_shared_args
from caiman_asr_train.args.mel_feat_norm import (
    add_mel_feat_norm_args,
    check_mel_feat_norm_args,
)
from caiman_asr_train.autotune.support import (
    MIN_TORCH_FOR_TILELANG,
    tilelang_supported,
)
from caiman_asr_train.utils.user_tokens_lite import get_all_user_tokens


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
        "--val_final_padding_secs",
        type=float,
        default=0.0,
        help="Validation audio will be padded at the end with this much silence",
    )
    parser.add_argument(
        "--val_manifests",
        type=str,
        required=True,
        nargs="+",
        help="Paths of the evaluation datasets manifest files",
    )
    parser.add_argument(
        "--val_dataset_dir",
        "--val_data_dir",
        type=str,
        help="Root dir of validation dataset",
        required=True,
    )
    parser.add_argument(
        "--val_standardizer_lang",
        type=str,
        required=True,
        help="ISO code of the language of the transcripts to be standardized",
    )
    parser.add_argument(
        "--inspect_audio",
        action="store_true",
        help="""Save audios (after augmentations are applied) to /results/augmented_audios.
        This will slow down the data loading process.""",
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
        "--loader_workers_per_gpu",
        type=int,
        help="Number of torch dataloading workers per GPU",
    )
    parser.add_argument(
        "--profiler",
        action="store_true",
        help="""Enable profiling with yappi and save top/nvidia-smi logs.
        This may slow down training/validation""",
    )
    parser.add_argument(
        "--calculate_emission_latency",
        action="store_true",
        default=False,
        help="Export timestamps per word to .ctm file and calculate emission latencies",
    )
    parser.add_argument(
        "--calculate_per",
        action="store_true",
        help="Calculate the Punctuation Error Rate (PER) during evaluation",
    )
    parser.add_argument(
        "--latency_head_offset",
        default=0.0,
        type=float,
        help="Estimate of the start-of-word timestamp latency",
    )
    parser.add_argument(
        "--latency_tail_offset",
        default=0.0,
        type=float,
        help="Estimate of the end-of-word timestamp latency",
    )
    parser.add_argument(
        "--allow_partial_checkpoint",
        action="store_true",
        default=False,
        help="Allow loading a checkpoint from an incomplete state dict",
    )
    parser.add_argument(
        "--transducer_backend",
        type=str,
        default="auto",
        choices=["auto", "cuda", "tilelang", "triton"],
        help="Backend for RNN-T Transducer Loss (default: %(default)s). "
        "In `auto` mode, uses tilelang if the GPU and container support it, "
        "otherwise falls back to triton.",
    )
    parser.add_argument(
        "--joint_lattice_cap",
        default=2**17,
        type=int,
        help="Cap on the lattice cells in one joint+loss split; tune this to balance "
        "the VRAM usage / throughput in the joint and loss. "
        "See docs/src/training/batch_size_hyperparameters.md",
    )
    add_state_reset_args(parser)
    add_mel_feat_norm_args(parser)
    add_per_args(parser)
    add_decoder_args(parser)
    add_eos_shared_args(parser)
    add_prefix_args(parser)


@beartype
def add_per_args(parser: ArgumentParser) -> None:
    per_args = parser.add_argument_group("PER analysis")
    per_args.add_argument(
        "--punctuation_marks",
        type=list,
        default=[",", ".", "?"],
        help="List of punctuation marks to consider for PER analysis",
    )
    per_args.add_argument(
        "--punctuation_normalization",
        type=bool,
        default=True,
        help="""Whether to normalize punctuation marks.
        When the punctuation error rate is calculated, the following
        punctuation marks are considered equivalent:

         `;` -> `,` semi-colon to comma
         `:` -> `.` colon to period
         `!` -> `.` exclamation mark to period
         `-` -> ` ` hyphen to space
        """,
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
def add_prefix_args(parser: ArgumentParser) -> None:
    prefix_args = parser.add_argument_group("Prefix Configuration")

    prefix_args.add_argument(
        "--pnc_prefix",
        type=str,
        choices=["pnc", "nopnc"],
        default=None,
        help="Prefix for inference/decoding, including train-time validations."
        "Choices: 'pnc' (<pnc>), 'nopnc' (<nopnc>). "
        "If not specified, no case/punctuation prefix is used.",
    )

    prefix_args.add_argument(
        "--val_lang_prefix",
        type=str,
        default=None,
        help="Language prefix to be used during inference/decoding. "
        "If not specified, no language prefix is used.",
    )


@beartype
def check_transducer_args(args: Namespace) -> None:
    if args.transducer_backend == "auto":
        args.transducer_backend = "tilelang" if tilelang_supported() else "triton"

    if args.transducer_backend == "tilelang":
        assert tilelang_supported(), (
            "--transducer_backend tilelang needs a CUDA GPU of compute capability "
            ">= 8.0 (Ampere and newer) for cp.async, and torch >= "
            f"{MIN_TORCH_FOR_TILELANG} for the device properties the autotuner "
            f"filters configs on; this container ships torch "
            f"{torch.__version__}. Use --transducer_backend triton."
        )

    if args.transducer_backend in ("triton", "tilelang"):
        with open(args.model_config, "r") as f:
            user_tokens, _ = get_all_user_tokens(yaml.safe_load(f))
        assert "star" not in user_tokens, (
            f"--transducer_backend {args.transducer_backend} does not implement the "
            f"star penalty, but {args.model_config} declares a star user token. "
            "Use --transducer_backend cuda."
        )

    # getattr: only the train parser has the delay-penalty args
    if (
        getattr(args, "dp_mu", 0.0) != 0.0
        or getattr(args, "dp_clamp", None) is not None
    ):
        assert args.transducer_backend in ("triton", "tilelang"), (
            "--dp_mu / --dp_clamp are only implemented by the triton and tilelang "
            f"transducer backends, got {args.transducer_backend!r}"
        )


@beartype
def check_shared_args(args: Namespace, *, check_transducer: bool = True) -> None:
    if args.fuzzy_topk_logits and args.decoder == "greedy":
        raise ValueError("--fuzzy_topk_logits is not supported with greedy decoding")

    if args.eos_is_terminal:
        assert args.eos_decoding == "predict", (
            "If --eos_is_terminal is set, --eos_decoding must be 'predict'. "
            "This is because the decoder cannot terminate on the EOS token if "
            "the model never predicts it."
        )

    if (
        args.calculate_emission_latency
        and args.decoder == "beam"
        and args.beam_no_partials
    ):
        raise ValueError(
            "Cannot calculate emission latency for beam search if --beam_no_partials is set"
        )

    if args.boost_keywords:
        assert (
            args.keywords_path is not None
        ), "Must provide --keywords_path when --boost_keywords is set"

    if check_transducer:
        check_transducer_args(args)

    check_mel_feat_norm_args(args)
